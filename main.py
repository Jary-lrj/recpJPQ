import os
import time
import torch
import argparse

from model import SASRec, GRU4Rec, NARM, SRGNN, STAMP
from utils import *
from test_embedding import visualize_embedding, plot_loss_curve
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity


def str2bool(s):
    if s not in {"false", "true"}:
        raise ValueError("Not a valid boolean string")
    return s == "true"


parser = argparse.ArgumentParser()
parser.add_argument("--model", default="SASRec", type=str)
parser.add_argument("--dataset", required=True)
parser.add_argument("--segment", default=8, type=int, required=True)
parser.add_argument("--type", default="normal", type=str, required=True)
parser.add_argument("--train_dir", required=True)
parser.add_argument("--batch_size", default=2048, type=int)
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--maxlen", default=50, type=int)
parser.add_argument("--hidden_units", default=200, type=int)
parser.add_argument("--num_blocks", default=2, type=int)
parser.add_argument("--num_epochs", default=200, type=int)
parser.add_argument("--num_heads", default=2, type=int)
parser.add_argument("--dropout_rate", default=0.2, type=float)
parser.add_argument("--l2_emb", default=0.0, type=float)
parser.add_argument("--device", default="cuda", type=str)
parser.add_argument("--inference_only", default=False, type=str2bool)
parser.add_argument("--state_dict_path", default=None, type=str)

args = parser.parse_args()
if not os.path.isdir(args.dataset + "_" + args.train_dir):
    os.makedirs(args.dataset + "_" + args.train_dir)
with open(os.path.join(args.dataset + "_" + args.train_dir, "args.txt"), "w") as f:
    f.write("\n".join([str(k) + "," + str(v)
            for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

if __name__ == "__main__":

    u2i_index, i2u_index = build_index(args.dataset)

    # global dataset
    dataset = data_partition(args.dataset)

    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    # num_batch = len(user_train) // args.batch_size # tail? + ((len(user_train) % args.batch_size) != 0)
    num_batch = (len(user_train) - 1) // args.batch_size + 1
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print("average sequence length: %.2f" % (cc / len(user_train)))

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(args.dataset + "_" +
                           args.train_dir, timestamp.split('-')[0])
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    f = open(
        os.path.join(
            log_dir, f"log_{timestamp}_{args.model}_{args.dataset}_{args.segment}_segment_{args.type}.txt"
        ),
        "w",
    )
    for arg in vars(args):
        f.write(f"{arg}: {getattr(args, arg)}\n")
    f.write("epoch (val_ndcg, val_hr) (test_ndcg, test_hr) loss\n")

    sampler = WarpSampler(
        user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3
    )
    if args.model == "SASRec":
        model = SASRec(usernum, itemnum, args).to(args.device)
    elif args.model == "GRU4Rec":
        model = GRU4Rec(usernum, itemnum, args).to(args.device)
    elif args.model == "NARM":
        model = NARM(usernum, itemnum, args).to(args.device)
    elif args.model == "SRGNN":
        model = SRGNN(usernum, itemnum, args).to(args.device)
    elif args.model == "STAMP":
        model = STAMP(usernum, itemnum, args).to(args.device)
    else:
        raise ValueError("Invalid model name")

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.numel()}, {param.device}")

    for name, param in model.named_parameters():
        try:
            # Initialize with a constant value of 0.02
            torch.nn.init.constant_(param.data, 0.02)
        except:
            pass  # just ignore those failed init layers

    # model.pos_emb.weight.data[0, :] = 0
    # model.item_emb.weight.data[0, :] = 0

    # this fails embedding init 'Embedding' object has no attribute 'dim'
    # model.apply(torch.nn.init.xavier_uniform_)

    # preprocessing for JPQ
    # codebook_t0 = time.time()
    # model.item_code.assign_codes_recJPQ(user_train)
    # codebook_t1 = time.time()
    # f.write(
    #     f"Time taken to build codebook: {codebook_t1 - codebook_t0:.2f} seconds\n")

    # preprocessing for DPQ
    initial_embedding = model.item_code.assign(user_train)
    codebook_t0 = time.time()
    model.item_code.assign_codes_KMeans(initial_embedding)
    codebook_t1 = time.time()
    f.write(
        f"Time taken to build codebook: {codebook_t1 - codebook_t0:.2f} seconds\n")

    # preprocessing for ours
    # codebook_t0 = time.time()
    # model.recat_build_codebook()
    # codebook_t1 = time.time()
    # f.write(
    #     f"Time taken to build codebook: {codebook_t1 - codebook_t0:.2f} seconds\n")

    model.train()  # enable model training

    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(
                args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find(
                "epoch=") + 6:]
            epoch_start_idx = int(tail[: tail.find(".")]) + 1
        except:  # in case your pytorch version is not 1.6 etc., pls debug by pdb if load weights failed
            print("failed loading state_dicts, pls check file path: ", end="")
            print(args.state_dict_path)
            print(
                "pdb enabled for your quick check, pls type exit() if you do not need it")
            import pdb

            pdb.set_trace()

    if args.inference_only:
        model.eval()
        t_test = evaluate(model, dataset, args)
        print("test (MRR@10: %.4f, NDCG@10: %.4f, HR@10: %.4f)" %
              (t_test[0], t_test[1], t_test[2]))

    # ce_criterion = torch.nn.CrossEntropyLoss()
    # https://github.com/NVIDIA/pix2pixHD/issues/9 how could an old bug appear again...
    bce_criterion = torch.nn.BCEWithLogitsLoss()  # torch.nn.BCELoss()
    adam_optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    best_val_ndcg, best_val_hr, best_val_mrr = 0.0, 0.0, 0.0
    best_test_ndcg, best_test_hr, best_test_mrr = 0.0, 0.0, 0.0
    HRs = []
    HRv = []
    NDCGs = []
    NDCGv = []
    T = 0.0
    t0 = time.time()

    loss_list = []

    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        if args.inference_only:
            break  # just to decrease identition
        avg_loss = 0.0
        for step in range(
            num_batch
        ):  # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            u, seq, pos, neg = sampler.next_batch()  # tuples to ndarray
            u, seq, pos, neg = np.array(u), np.array(
                seq), np.array(pos), np.array(neg)
            pos_logits, neg_logits = model(u, seq, pos, neg)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(
                neg_logits.shape, device=args.device
            )
            # print("\neye ball check raw_logits:"); print(pos_logits); print(neg_logits) # check pos_logits > 0, neg_logits < 0
            adam_optimizer.zero_grad()
            indices = np.where(pos != 0)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            for param in model.item_code.parameters():
                loss += args.l2_emb * torch.norm(param)
            loss.backward()
            adam_optimizer.step()
            avg_loss += loss.item()  # avg loss in each epoch

        avg_loss /= num_batch
        print(f"avg loss in epoch {epoch}: {avg_loss}")
        loss_list.append(avg_loss)

        if epoch % 20 == 0:
            model.eval()
            t1 = time.time() - t0
            T += t1
            print("Evaluating", end="")
            # t_train = evaluate_all(model, dataset, args, "train")
            t_test = evaluate(model, dataset, args)
            t_valid = evaluate_valid(model, dataset, args)
            # print(
            #     "epoch:%d, time: %f(s), train (NDCG@10: %.4f, HR@10: %.4f), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)"
            #     % (epoch, T, t_train[0], t_train[1], t_valid[0], t_valid[1], t_test[0], t_test[1])
            # )
            print("-" * 50)
            print(f"Epoch: {epoch}")
            print(f"Time Taken: {T:.2f} seconds")
            print("Validation Metrics:")
            print(f"  - MRR@10:  {t_valid[0]:.4f}")
            print(f"  - NDCG@10: {t_valid[1]:.4f}")
            print(f"  - HR@10:   {t_valid[2]:.4f}")
            print("Test Metrics:")
            print(f"  - MRR@10:  {t_test[0]:.4f}")
            print(f"  - NDCG@10: {t_test[1]:.4f}")
            print(f"  - HR@10:   {t_test[2]:.4f}")
            print("-" * 50)
            NDCGv.append(t_valid[1])
            NDCGs.append(t_test[1])
            HRv.append(t_valid[2])
            HRs.append(t_test[2])
            if (
                t_valid[0] > best_val_mrr
                or t_valid[1] > best_val_ndcg
                or t_valid[2] > best_val_hr
                or t_test[0] > best_test_mrr
                or t_test[1] > best_test_ndcg
                or t_test[2] > best_test_hr
            ):
                best_val_mrr = max(t_valid[0], best_val_mrr)
                best_val_ndcg = max(t_valid[1], best_val_ndcg)
                best_val_hr = max(t_valid[2], best_val_hr)
                best_test_mrr = max(t_test[0], best_test_mrr)
                best_test_ndcg = max(t_test[1], best_test_ndcg)
                best_test_hr = max(t_test[2], best_test_hr)

            print("-" * 50)
            print("Current Best Results:")
            print(f"  Validation Metrics:")
            print(f"    - MRR:  {best_val_mrr:.4f}")
            print(f"    - NDCG: {best_val_ndcg:.4f}")
            print(f"    - HR:   {best_val_hr:.4f}")
            print(f"  Test Metrics:")
            print(f"    - MRR:  {best_test_mrr:.4f}")
            print(f"    - NDCG: {best_test_ndcg:.4f}")
            print(f"    - HR:   {best_test_hr:.4f}")
            print("-" * 50)
            f.write(str(epoch) + " " + str(t_valid) + " " +
                    str(t_test) + str(avg_loss) + "\n")
            f.flush()
            t0 = time.time()
            model.train()

        if epoch == args.num_epochs:
            f.write("best results\n")
            f.write(f"Validation Metrics:\n")
            f.write(f"  - MRR:  {best_val_mrr:.4f}\n")
            f.write(f"  - NDCG: {best_val_ndcg:.4f}\n")
            f.write(f"  - HR:   {best_val_hr:.4f}\n")
            f.write(f"Test Metrics:\n")
            f.write(f"  - MRR:  {best_test_mrr:.4f}\n")
            f.write(f"  - NDCG: {best_test_ndcg:.4f}\n")
            f.write(f"  - HR:   {best_test_hr:.4f}\n")
            max_ndcg_v_index = NDCGv.index(max(NDCGv))
            max_hr_v_index = HRv.index(max(HRv))
            corresponding_ndcg_s = NDCGs[max_ndcg_v_index]
            corresponding_hr_s = HRs[max_hr_v_index]
            f.write(
                f"Corresponding test NDCG by max valid NDCG: {corresponding_ndcg_s:.4f}\n")
            f.write(
                f"Corresponding test HR by max valid HR: {corresponding_hr_s:.4f}\n")
            f.write(f"Average NDCG: {sum(NDCGs) / len(NDCGs):.4f}\n")
            f.write(f"Average HR: {sum(HRs) / len(HRs):.4f}\n")
            f.write(f"Time Per Epoch: {T / epoch:.2f} seconds\n")
            folder = log_dir
            fname = "final_{}.type={}.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}_{}.pth"
            fname = fname.format(
                args.model,
                args.type,
                epoch,
                args.lr,
                args.num_blocks,
                args.num_heads,
                args.hidden_units,
                args.maxlen,
                timestamp,
            )
            torch.save(model.state_dict(), os.path.join(folder, fname))
            # when dealing with baseline, we need to save the embeddings
            try:
                if args.type == "base":
                    item_embeddings = model.item_emb.weight
                elif args.type == "QR":
                    item_embeddings = model.get_all_item_embeddings()
                else:
                    item_embeddings = model.item_code.get_all_item_embeddings()

                similarity_matrix = cosine_similarity(
                    item_embeddings.detach().cpu().numpy())
                np.fill_diagonal(similarity_matrix, 0)
                avg_nearest_neighbor_similarity = np.mean(
                    np.max(similarity_matrix, axis=1))
                f.write(
                    f"CECS: {avg_nearest_neighbor_similarity:.4}\n")
                f.write(f"Time per epoch: {T / epoch:.2f} seconds")
                print("Visualizing Embeddings...")
                visualize_embedding(
                    "euclidean",
                    item_embeddings,
                    output_filename=os.path.join(
                        log_dir,
                        f"{args.dataset}_{args.model}_{args.segment}_segment_{args.type}_{timestamp}.png",
                    ),
                    figsize=(20, 15),
                    dpi=300,
                    save_statistics=True,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    segment_size=50,
                )
                print("Visulize Done.")
            except Exception as e:
                print("Failed. Error: ", e)

    f.close()
    sampler.close()
    print("Done")

    plot_loss_curve(args.model, loss_list, args.dataset,
                    args.segment, args.type)
