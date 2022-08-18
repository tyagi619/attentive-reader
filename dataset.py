import torch
import torch.nn.functional as F
import torch.utils.data as data

import numpy as np


class SQuAD(data.Dataset):
    def __init__(self, data_path, use_v2=True):
        super(SQuAD, self).__init__()

        dataset = np.load(data_path)
        self.context_idxs = torch.from_numpy(dataset['context_idxs']).long()
        self.context_char_idxs = torch.from_numpy(dataset['context_char_idxs']).long()
        self.question_idxs = torch.from_numpy(dataset['ques_idxs']).long()
        self.question_char_idxs = torch.from_numpy(dataset['ques_char_idxs']).long()
        self.y1s = torch.from_numpy(dataset['y1s']).long()
        self.y2s = torch.from_numpy(dataset['y2s']).long()

        if use_v2:
            # If using SQuAD 2.0 (non-answerable questions), start
            # each sentence with OOV (token 1) and set answer start
            # and end to 0th index (OOV token)

            # TODO : instead of OOV use NULL (token 0) for non-answerable
            # questions. Idea is that NULL tokens would be better understood
            # by the model since OOV words will also occur within the context
            # Possible problem : NULL is a pad token so it might cause issues
            # Possible Fix : Set up another token for this case
            batch_size, _, w_len = self.context_char_idxs.size()
            
            ones = torch.ones((batch_size, 1), dtype=torch.int64)
            self.context_idxs = torch.cat((ones, self.context_idxs), dim=1)
            self.question_idxs = torch.cat((ones, self.question_idxs), dim=1)

            ones = torch.ones((batch_size, 1, w_len), dtype=torch.int64)
            self.context_char_idxs = torch.cat((ones, self.context_char_idxs), dim=1)
            self.question_char_idxs = torch.cat((ones, self.question_char_idxs), dim=1)

            self.y1s += 1
            self.y2s += 1

        self.ids = torch.from_numpy(dataset['ids']).long()
        self.valid_idxs = [idx for idx in range(len(self.ids))
                           if use_v2 or self.y1s[idx].item() >= 0]

    def __getitem__(self, idx):
        idx = self.valid_idxs[idx]
        example = (self.context_idxs[idx],
                   self.context_char_idxs[idx],
                   self.question_idxs[idx],
                   self.question_char_idxs[idx],
                   self.y1s[idx],
                   self.y2s[idx],
                   self.ids[idx])
        return example

    def __len__(self):
        return len(self.valid_idxs)
