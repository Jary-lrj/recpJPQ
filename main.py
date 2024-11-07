import os
import time
import torch
import argparse

from model import SASRec
from utils import *


def str2bool(s):
    if s not in {"false", "true"}:
        raise ValueError("Not a valid boolean string")
    return s == "true"


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--train_dir", required=True)
parser.add_argument("--batch_size", default=256, type=int)
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--maxlen", default=200, type=int)
parser.add_argument("--hidden_units", default=400, type=int)
parser.add_argument("--num_blocks", default=2, type=int)
parser.add_argument("--num_epochs", default=200, type=int)
parser.add_argument("--num_heads", default=1, type=int)
parser.add_argument("--dropout_rate", default=0.2, type=float)
parser.add_argument("--l2_emb", default=0.0, type=float)
parser.add_argument("--device", default="cuda", type=str)
parser.add_argument("--inference_only", default=False, type=str2bool)
parser.add_argument("--state_dict_path", default=None, type=str)

args = parser.parse_args()
if not os.path.isdir(args.dataset + "_" + args.train_dir):
    os.makedirs(args.dataset + "_" + args.train_dir)
with open(os.path.join(args.dataset + "_" + args.train_dir, "args.txt"), "w") as f:
    f.write("\n".join([str(k) + "," + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
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

    f = open(os.path.join(args.dataset + "_" + args.train_dir, "log.txt"), "w")
    f.write("epoch (val_ndcg, val_hr) (test_ndcg, test_hr)\n")

    sampler = WarpSampler(
        user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3
    )
    model = SASRec(usernum, itemnum, args).to(
        args.device
    )  # no ReLU activation in original SASRec implementation?

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.numel()}")

    # initialize item code
    model.item_code.assign_codes(user_train)

    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass  # just ignore those failed init layers

    model.pos_emb.weight.data[0, :] = 0
    model.item_emb.weight.data[0, :] = 0

    # this fails embedding init 'Embedding' object has no attribute 'dim'
    # model.apply(torch.nn.init.xavier_uniform_)

    model.train()  # enable model training

    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find("epoch=") + 6 :]
            epoch_start_idx = int(tail[: tail.find(".")]) + 1
        except:  # in case your pytorch version is not 1.6 etc., pls debug by pdb if load weights failed
            print("failed loading state_dicts, pls check file path: ", end="")
            print(args.state_dict_path)
            print("pdb enabled for your quick check, pls type exit() if you do not need it")
            import pdb

            pdb.set_trace()

    if args.inference_only:
        model.eval()
        t_test = evaluate(model, dataset, args)
        print("test (NDCG@10: %.4f, HR@10: %.4f)" % (t_test[0], t_test[1]))

    # ce_criterion = torch.nn.CrossEntropyLoss()
    # https://github.com/NVIDIA/pix2pixHD/issues/9 how could an old bug appear again...
    bce_criterion = torch.nn.BCEWithLogitsLoss()  # torch.nn.BCELoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    best_val_ndcg, best_val_hr = 0.0, 0.0
    best_test_ndcg, best_test_hr = 0.0, 0.0
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
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
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
            print(
                "epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)"
                % (epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1])
            )

            if (
                t_valid[0] > best_val_ndcg
                or t_valid[1] > best_val_hr
                or t_test[0] > best_test_ndcg
                or t_test[1] > best_test_hr
            ):
                best_val_ndcg = max(t_valid[0], best_val_ndcg)
                best_val_hr = max(t_valid[1], best_val_hr)
                best_test_ndcg = max(t_test[0], best_test_ndcg)
                best_test_hr = max(t_test[1], best_test_hr)
                folder = args.dataset + "_" + args.train_dir
                fname = "SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth"
                fname = fname.format(
                    epoch, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen
                )
                torch.save(model.state_dict(), os.path.join(folder, fname))
            print(
                f"Current best result: value_NDCG {best_val_ndcg}, value_HR {best_val_hr}, test_NDCG {best_test_ndcg}, test_HR {best_test_hr}"
            )
            f.write(str(epoch) + " " + str(t_valid) + " " + str(t_test) + "\n")
            f.flush()
            t0 = time.time()
            model.train()

        if epoch == args.num_epochs:
            folder = args.dataset + "_" + args.train_dir
            fname = "SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth"
            fname = fname.format(
                args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen
            )
            torch.save(model.state_dict(), os.path.join(folder, fname))
            # save item embeddings
            item_embeddings = []
            for i in range(itemnum):
                item_embeddings.append(model.item_code.get_item_embedding(i).to("cpu"))
            item_embeddings.append(torch.zeros(args.hidden_units))
            item_embeddings_torch = torch.stack(item_embeddings, dim=0)
            torch.save(item_embeddings_torch, os.path.join("item_embeddings.pth"))

    f.close()
    sampler.close()
    print("Done")

    import matplotlib.pyplot as plt

    plt.plot(loss_list)
    plt.savefig("loss.png")