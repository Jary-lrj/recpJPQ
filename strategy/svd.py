import torch
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.decomposition import TruncatedSVD
from .centroid_strategy import CentroidAssignmentStragety


class SVDAssignmentStrategy(CentroidAssignmentStragety):
    def assign(self, train_users):
        rows = []
        cols = []
        vals = []
        for user, item_set in train_users.items():
            for item in item_set:
                rows.append(user - 1)
                cols.append(item)
                vals.append(1)

        # Convert to PyTorch sparse tensor
        indices = torch.LongTensor([rows, cols]).to(self.device)
        values = torch.FloatTensor(vals).to(self.device)
        shape = (len(train_users), self.num_items + 1)
        matr = torch.sparse_coo_tensor(indices, values, shape).to(self.device)

        print("fitting svd for initial centroids assignments")
        svd = TruncatedSVD(n_components=self.item_code_bytes)
        svd.fit(matr.cpu().to_dense().numpy())
        item_embeddings = torch.from_numpy(svd.components_).to(self.device)

        assignments = []
        print("done")

        for i in range(self.item_code_bytes):
            discretizer = KBinsDiscretizer(n_bins=256, encode="ordinal", strategy="quantile")
            ith_component = item_embeddings[i : i + 1][0]
            ith_component = (ith_component - ith_component.min()) / (
                ith_component.max() - ith_component.min() + 1e-10
            )

            noise = torch.randn(self.num_items + 1, device=self.device) * 1e-5
            ith_component += noise  # make sure that every item has unique value

            ith_component = ith_component.unsqueeze(1).cpu().numpy()
            component_assignments = discretizer.fit_transform(ith_component).astype("uint8")[:, 0]
            assignments.append(torch.from_numpy(component_assignments).to(self.device))

        final_tensor = torch.stack(assignments).t()
        print("KBinsDiscretizer done")
        return final_tensor
