import torch
from torch.utils.data import Dataset, DataLoader
EOS_token = 1

class dataset(Dataset):
    def __init__(self, input_lang, output_lang, pairs):
        super().__init__()
        self.input_lang = input_lang
        self.output_lang = output_lang
        self.pairs = pairs
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, index):
        return self.tensorsFromPair(self.pairs[index])

    def indexesFromSentence(self, lang, sentence):
        return [lang.word2index[word] for word in sentence.split(' ')]

    def tensorFromSentence(self, lang, sentence):
        indexes = self.indexesFromSentence(lang, sentence)
        indexes.append(EOS_token)
        return torch.tensor(indexes, dtype=torch.long)

    def tensorsFromPair(self, pair):
        input_tensor = self.tensorFromSentence(self.input_lang, pair[0])
        target_tensor = self.tensorFromSentence(self.output_lang, pair[1])
        return input_tensor, target_tensor

def custom_collate(batch):
    batch_size = len(batch)
    batch = sorted(batch, key = lambda sample : sample[0].size(0))
    seq_length = [sample[0].size(0) for sample in batch]
    target_length = [sample[1].size(0) for sample in batch]
    max_seq_length = max(seq_length)
    max_target_length = max(target_length)

    seqs = torch.zeros((batch_size, max_seq_length)).to(torch.long)
    targets = torch.zeros((batch_size, max_target_length)).to(torch.long)

    for i in range(0, batch_size):
        seq, target = batch[i]
        
        seqs[i].narrow(0, 0, seq.size(0)).copy_(seq)
        targets[i].narrow(0, 0, target.size(0)).copy_(target)

    seq_length = torch.IntTensor(seq_length)
    target_length = torch.IntTensor(target_length)
    return seqs, targets, seq_length, target_length

class dataloader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.collate_fn = custom_collate