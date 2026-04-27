import torch
import os
import numpy as np
import sys
import pysam
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import AutoConfig, AutoModelForMaskedLM
from torcheval.metrics import BinaryAccuracy, BinaryAUROC, BinaryF1Score, BinaryAUPRC

from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist



class ADDataModule():
    def __init__(self, data_dir: str = "./Dataset/ADdatasets.csv",vali_set = ["chr10"], test_set=["chr8","chr9"], seq_length = 4096,batch_size: int = 64, pretrain = False):
        super().__init__()

        print(os.getcwd())

        faste_path = "/projectnb/ds596/students/alicehan/EpiModX/Dataset/hg38.fa"
        self.fasta_file =pysam.Fastafile(faste_path)
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.vali_set = vali_set
        self.test_set = test_set
        self.seq_length = seq_length
        df_data = pd.read_csv(self.data_dir)
        self.pretrain = pretrain
        
        self.train_data =HistoneDataset(df_data[(~df_data["chrom"].isin(self.vali_set))&(~df_data["chrom"].isin(self.test_set))],self.fasta_file, self.seq_length, self.pretrain)
        self.vali_data =  HistoneDataset(df_data[df_data["chrom"].isin(self.vali_set)],self.fasta_file, self.seq_length,self.pretrain)
        self.test_data = HistoneDataset(df_data[df_data["chrom"].isin(self.test_set)],self.fasta_file, self.seq_length,self.pretrain)

    def train_dataloader(self):
        # return DataLoader(self.train_data, batch_size=self.batch_size,shuffle=True)
        if dist.is_initialized():
            sampler = DistributedSampler(self.train_data)
            return DataLoader(self.train_data, batch_size=self.batch_size, sampler=sampler)
        else:
            return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        # return DataLoader(self.vali_data, batch_size=self.batch_size,shuffle=False)
        if dist.is_initialized():
            sampler = DistributedSampler(self.vali_data, shuffle=False)
            return DataLoader(
                self.vali_data,
                batch_size=self.batch_size,
                sampler=sampler,
                num_workers=4
            )
        else:
            return DataLoader(
                self.vali_data,
                batch_size=self.batch_size,
                shuffle=False
            )

    def test_dataloader(self):
        # return DataLoader(self.test_data, batch_size=self.batch_size,shuffle=False)
        if dist.is_initialized():
            sampler = DistributedSampler(self.test_data, shuffle=False)
            return DataLoader(
                self.test_data,
                batch_size=self.batch_size,
                sampler=sampler,
                num_workers=4
            )
        else:
            return DataLoader(
                self.test_data,
                batch_size=self.batch_size,
                shuffle=False
            )


class HistoneDataset(Dataset):
    def __init__(self, datafiles, fasta_file,seq_length, pretrain = False, transform=None, target_transform=None):
        self.data_file = datafiles
        self.fasta_file = fasta_file
        self.seq_length = seq_length
        self.pretrain = pretrain
        if self.pretrain:
            model_name = "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    def __len__(self):
        return self.data_file.shape[0]

    def __getitem__(self, idx):
        chr_temp, start_chr, end_chr = self.data_file.iloc[idx,:3]
        label = self.data_file.iloc[idx,3:]
        start_position = int((start_chr+end_chr)/2)-int(self.seq_length/2)
        seq = self.fasta_file.fetch(chr_temp,start_position, start_position+self.seq_length)
        if self.pretrain:
            encoded_sequence = self.tokenizer(seq, return_tensors="pt")['input_ids'][0]
        else:
            encoded_sequence = one_hot_encode_dna(seq)

        sample = {'sequence': encoded_sequence,
                 'label': torch.tensor(label.to_list(),dtype=torch.float)}
        # sample.update(label.to_dict())
        
        return  sample

class mutationDataset(Dataset):
    def __init__(self,temp_chr, snp,seq_length, test_length,pretrain = False, transform=None, target_transform=None):
        self.snp = snp
        self.chr = temp_chr
        # faste_path = "/home/xiaoyu/Genome/data/human/genome/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta"
        faste_path = "/projectnb/ds596/students/alicehan/EpiModX/Dataset/hg38.fa"
        self.fasta_file =pysam.Fastafile(faste_path)
        self.seq_length = seq_length
        self.pretrain = pretrain
        self.test_length = test_length
        self.mutation = ["A","G","C","T"]
        if self.pretrain:
            model_name = "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    def __len__(self):
        return len(self.snp)*4*self.test_length

    def __getitem__(self, idx):
        snp_idx = idx//(self.test_length*4)
        pos_idx = (idx-snp_idx*(self.test_length*4))%self.test_length
        new_mutation = self.mutation[(idx-snp_idx*(self.test_length*4))//self.test_length]
        start_chr, end_chr = int(self.snp[snp_idx]-self.seq_length/2), int(self.snp[snp_idx]+self.seq_length/2)
        seq = self.fasta_file.fetch(self.chr,start_chr, end_chr)
        loc = int(self.seq_length/2-self.test_length/2+pos_idx)
        new_seq = seq[:loc] + new_mutation + seq[loc+1:]
        # print(loc, new_mutation,start_chr, end_chr)
       
        if self.pretrain:
            encoded_sequence = self.tokenizer(new_seq, return_tensors="pt")['input_ids'][0]
        else:
            encoded_sequence = one_hot_encode_dna(new_seq)
        
        return  encoded_sequence

def one_hot_encode_dna(seq):
    seq_len = len(seq)
    seq = seq.upper()
    base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    seq_code = torch.zeros((seq_len, 4))

    for i, base in enumerate(seq):
        if base in base_map:
            index = base_map[base]
            seq_code[i, index] = 1
    
    return seq_code.transpose(0, 1)

class multiperformance():

    def __init__(self):
        self.metric_dict = { "accuracy": BinaryAccuracy(),"AUC":BinaryAUROC(),"F1":BinaryF1Score(),"PRC": BinaryAUPRC()}

    def update(self, inputs, target):
        for name, m in self.metric_dict.items():
            m.update(inputs, target)

    def compute(self):
        return { name:m.compute() for name,m in self.metric_dict.items()}

    def reset(self):
        for name, m in self.metric_dict.items():
            m.reset()


class AMLDataModule():
    def __init__(
        self,
        data_path,
        vali_set=["chr10"],
        test_set=["chr8", "chr9"],
        seq_length=4096,
        batch_size=64,
        pretrain=False,
        num_labels=22,
        label_mode="balanced"  # "first", "aml", "healthy", "balanced"
    ):
        super().__init__()

        print(os.getcwd())

        fasta_path = "/projectnb/ds596/students/alicehan/EpiModX/Dataset/hg38.fa"
        self.fasta_file = pysam.Fastafile(fasta_path)

        self.batch_size = batch_size
        self.vali_set = vali_set
        self.test_set = test_set
        self.seq_length = seq_length
        self.pretrain = pretrain

        df = pd.read_csv(data_path)

        # --- Identify label columns ---
        all_label_cols = df.columns[3:]

        aml_cols = [c for c in all_label_cols if "AML" in c]
        healthy_cols = [c for c in all_label_cols if "Healthy" in c]

        if label_mode == "first":
            selected_cols = list(all_label_cols[:num_labels])

        elif label_mode == "aml":
            selected_cols = aml_cols[:num_labels]

        elif label_mode == "healthy":
            selected_cols = healthy_cols[:num_labels]

        elif label_mode == "balanced":
            half = num_labels // 2
            selected_cols = aml_cols[:half] + healthy_cols[:half]

        else:
            raise ValueError("Invalid label_mode")

        print("Selected label columns:", selected_cols)
        print("Total labels:", len(selected_cols))

        self.label_cols = selected_cols

        # --- Split dataset ---
        train_df = df[(~df["chrom"].isin(vali_set)) & (~df["chrom"].isin(test_set))]
        val_df = df[df["chrom"].isin(vali_set)]
        test_df = df[df["chrom"].isin(test_set)]

        self.train_data = AMLDataset(train_df, self.fasta_file, seq_length, pretrain, self.label_cols)
        self.vali_data = AMLDataset(val_df, self.fasta_file, seq_length, pretrain, self.label_cols)
        self.test_data = AMLDataset(test_df, self.fasta_file, seq_length, pretrain, self.label_cols)

    def train_dataloader(self):
        if dist.is_initialized():
            sampler = DistributedSampler(self.train_data)
            return DataLoader(self.train_data, batch_size=self.batch_size, sampler=sampler, num_workers=6)
        else:
            return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=6)

    def val_dataloader(self):
        if dist.is_initialized():
            sampler = DistributedSampler(self.vali_data, shuffle=False)
            return DataLoader(self.vali_data, batch_size=self.batch_size, sampler=sampler, num_workers=6)
        else:
            return DataLoader(self.vali_data, batch_size=self.batch_size, shuffle=False, num_workers=6)

    def test_dataloader(self):
        if dist.is_initialized():
            sampler = DistributedSampler(self.test_data, shuffle=False)
            return DataLoader(self.test_data, batch_size=self.batch_size, sampler=sampler, num_workers=6)
        else:
            return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=6)

class AMLDataset(Dataset):
    def __init__(self, df, fasta_file, seq_length, pretrain, label_cols):
        self.df = df.reset_index(drop=True)
        self.fasta_file = fasta_file
        self.seq_length = seq_length
        self.pretrain = pretrain
        self.label_cols = label_cols

        if self.pretrain:
            model_name = "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        chr_temp = row["chrom"]
        start_chr = row["start"]
        end_chr = row["end"]

        # --- sequence ---
        center = int((start_chr + end_chr) / 2)
        start_position = center - int(self.seq_length / 2)

        seq = self.fasta_file.fetch(chr_temp, start_position, start_position + self.seq_length)

        if self.pretrain:
            encoded_sequence = self.tokenizer(seq, return_tensors="pt")["input_ids"][0]
        else:
            encoded_sequence = one_hot_encode_dna(seq)

        # --- labels ---
        # label = row[self.label_cols].values.astype(float)
        row = self.df.iloc[idx]

        aml_cols = [c for c in self.df.columns if "AML" in c]
        healthy_cols = [c for c in self.df.columns if "Healthy" in c]

        aml_signal = row[aml_cols].mean()
        healthy_signal = row[healthy_cols].mean()

        label = 1.0 if aml_signal > healthy_signal else 0.0

        return {
            "sequence": encoded_sequence,
            "label": torch.tensor([label], dtype=torch.float)
        }

