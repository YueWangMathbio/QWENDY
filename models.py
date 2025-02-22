"""
models of TEQWENDY and LEQWENDY methods
"""

import torch
import torch.nn as nn
import numpy as np
from transformers import RobertaModel, RobertaConfig
from peft import LoraConfig, get_peft_model, TaskType

class RobertaEncoderWrapper(nn.Module): 
    # wrap the encoder of Roberta, so that LoRA can work
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder  
    def forward(self, inputs_embeds, attention_mask, **kwargs):
        if attention_mask.dtype != torch.bool:
            attention_mask = attention_mask.to(dtype=inputs_embeds.dtype)  # Match dtype        
        return self.encoder(hidden_states=inputs_embeds, attention_mask=attention_mask).last_hidden_state



class Leqwendy_first_half(nn.Module):
    # the first half of LEQWENDY model
    # input size: batch_size * 4 * height * width
    # output size: batch_size * 4 * height * width
    def __init__(self, lora_r=8, lora_alpha=16):
        super(Leqwendy_first_half, self).__init__()
        
        self.embedding_k0 = nn.Sequential(nn.Linear(1, 256), nn.ReLU(), nn.Linear(256, 256))
        self.embedding_k1 = nn.Sequential(nn.Linear(1, 256), nn.ReLU(), nn.Linear(256, 256))
        self.embedding_k2 = nn.Sequential(nn.Linear(1, 256), nn.ReLU(), nn.Linear(256, 256))
        self.embedding_k3 = nn.Sequential(nn.Linear(1, 256), nn.ReLU(), nn.Linear(256, 256))

        self.segment_embedding = nn.Embedding(4, 256)  
        
        model_path = "weights"
        config = RobertaConfig.from_pretrained(model_path)
        roberta_model = RobertaModel.from_pretrained(model_path, config=config)
        roberta_encoder = RobertaEncoderWrapper(roberta_model.encoder)
        for param in roberta_encoder.parameters():
            param.requires_grad = False
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,  
            r=lora_r, 
            lora_alpha=lora_alpha, 
            lora_dropout=0.1,             
            target_modules=["query", "value"]  
        )
        roberta_encoder = get_peft_model(roberta_encoder, lora_config)
        self.transformer_encoder = roberta_encoder
        
        self.output_k1l = nn.Sequential(nn.Linear(1024, 512), nn.LeakyReLU(0.1), nn.Linear(512, 1))
        self.output_k2l = nn.Sequential(nn.Linear(1024, 512), nn.LeakyReLU(0.1), nn.Linear(512, 1))
        self.output_k3l = nn.Sequential(nn.Linear(1024, 512), nn.LeakyReLU(0.1), nn.Linear(512, 1))

    def positional_encoding(self, height, width, device):
        
        div_term = torch.exp(
            torch.arange(0, 128, 2, device=device, dtype=torch.float) * 
            (-np.log(10000.0) / 128)
        )
        y_position = torch.arange(height, dtype=torch.float, device=device).unsqueeze(1) 
        x_position = torch.arange(width, dtype=torch.float, device=device).unsqueeze(1)  
        pe_y = torch.zeros(height, 128, device=device)
        pe_x = torch.zeros(width, 128, device=device)
        
        pe_y[:, 0::2] = torch.sin(y_position * div_term)
        pe_y[:, 1::2] = torch.cos(y_position * div_term)
        
        pe_x[:, 0::2] = torch.sin(x_position * div_term)
        pe_x[:, 1::2] = torch.cos(x_position * div_term)
        
        pe_y_expanded = pe_y.unsqueeze(1).expand(height, width, 128) 
        pe_x_expanded = pe_x.unsqueeze(0).expand(height, width, 128) 
        pe = torch.cat([pe_y_expanded, pe_x_expanded], dim=-1)
        pe = pe.repeat(1, 1, 4)
        return pe
    
    def forward(self, k_all):
        device = k_all.device  
        k0 = k_all[:, 0]
        k1 = k_all[:, 1]
        k2 = k_all[:, 2]
        k3 = k_all[:, 3]
        batch_size, height, width = k0.shape
        k0_embedded = self.embedding_k0(k0.unsqueeze(-1))
        k1_embedded = self.embedding_k1(k1.unsqueeze(-1))
        k2_embedded = self.embedding_k2(k2.unsqueeze(-1))
        k3_embedded = self.embedding_k3(k3.unsqueeze(-1))
        
        segment_ids = 0 * torch.ones_like(k0, dtype=torch.long, device=device)
        segment_embedding_0 = self.segment_embedding(segment_ids)
        segment_ids = 1 * torch.ones_like(k0, dtype=torch.long, device=device)
        segment_embedding_1 = self.segment_embedding(segment_ids)
        segment_ids = 2 * torch.ones_like(k0, dtype=torch.long, device=device)
        segment_embedding_2 = self.segment_embedding(segment_ids)
        segment_ids = 3 * torch.ones_like(k0, dtype=torch.long, device=device)
        segment_embedding_3 = self.segment_embedding(segment_ids)
        
        k0_embedded += segment_embedding_0
        k1_embedded += segment_embedding_1
        k2_embedded += segment_embedding_2
        k3_embedded += segment_embedding_3

        combined_embedding = torch.cat((k0_embedded, k1_embedded, k2_embedded, k3_embedded), dim=-1)  # Shape: (batch_size, height, width, 1024)

        pe = self.positional_encoding(height, width, device)
        src = combined_embedding + pe
        src = src.view(batch_size, height * width, -1).contiguous()

        attention_mask = torch.ones((batch_size, src.shape[1]), dtype=torch.long, device=device)
        attention_mask = attention_mask[:, None, None, :]

        output = self.transformer_encoder(inputs_embeds=src, attention_mask=attention_mask)

        output_k1l = self.output_k1l(output) 
        output_k2l = self.output_k2l(output)
        output_k3l = self.output_k3l(output)
        output_k1l = output_k1l.view(batch_size, 1, height, width)
        output_k2l = output_k2l.view(batch_size, 1, height, width)
        output_k3l = output_k3l.view(batch_size, 1, height, width)
        output = torch.cat([k0.unsqueeze(1), output_k1l, output_k2l, output_k3l], dim=1)
        return output


class Leqwendy_second_half(nn.Module): 
    # the second half of LEQWENDY model
    # input size: batch_size * 5 * height * width
    # output size: batch_size * height * width
    def __init__(self, lora_r=16, lora_alpha=32):
        super(Leqwendy_second_half, self).__init__()
        self.d_model_1 = 192
        self.d_model_2 = 256
        self.embedding = nn.ModuleList([])
        for i in range(4):
            self.embedding.append(nn.Sequential(nn.Linear(1, self.d_model_1), nn.ReLU(), nn.Linear(self.d_model_1, self.d_model_1)))
        self.embedding.append(nn.Sequential(nn.Linear(1, self.d_model_2), nn.ReLU(), nn.Linear(self.d_model_2, self.d_model_2)))
        self.segment_embedding = nn.Embedding(4, self.d_model_1)  
        model_path = "weights"
        config = RobertaConfig.from_pretrained(model_path)
        roberta_model = RobertaModel.from_pretrained(model_path, config=config)
        roberta_encoder = RobertaEncoderWrapper(roberta_model.encoder)
        for param in roberta_encoder.parameters():
            param.requires_grad = False
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,  
            r=lora_r, 
            lora_alpha=lora_alpha, 
            lora_dropout=0.1,             
            target_modules=["query", "value"]  
        )
        roberta_encoder = get_peft_model(roberta_encoder, lora_config)
        self.transformer_encoder = roberta_encoder
        self.output = nn.Sequential(nn.Linear(1024, 1024), nn.LeakyReLU(0.1), nn.Linear(1024, 1))

    def positional_encoding(self, height, width, dim, device): 
        halfdim = dim // 2
        div_term = torch.exp(
            torch.arange(0, halfdim, 2, device=device, dtype=torch.float) * 
            (-np.log(10000.0) / halfdim)
        )
        y_position = torch.arange(height, dtype=torch.float, device=device).unsqueeze(1) 
        x_position = torch.arange(width, dtype=torch.float, device=device).unsqueeze(1)  
        pe_y = torch.zeros(height, halfdim, device=device)
        pe_x = torch.zeros(width, halfdim, device=device)        
        pe_y[:, 0::2] = torch.sin(y_position * div_term)
        pe_y[:, 1::2] = torch.cos(y_position * div_term)        
        pe_x[:, 0::2] = torch.sin(x_position * div_term)
        pe_x[:, 1::2] = torch.cos(x_position * div_term)        
        pe_y_expanded = pe_y.unsqueeze(1).expand(height, width, halfdim) 
        pe_x_expanded = pe_x.unsqueeze(0).expand(height, width, halfdim) 
        pe = torch.cat([pe_y_expanded, pe_x_expanded], dim=-1)
        return pe

    def forward(self, K):
        device = K.device  
        batch_size, height, width = K[:, 0].shape
        K_embedded = []
        pe = self.positional_encoding(height, width, self.d_model_1, device)
        for i in range(4):
            temp = self.embedding[i](K[:, i].unsqueeze(-1))
            segment_ids = i * torch.ones_like(K[:, 0], dtype=torch.long, device=device)
            se = self.segment_embedding(segment_ids)
            temp += pe + se
            K_embedded.append(temp)
        temp = self.embedding[4](K[:, 4].unsqueeze(-1))
        pe = self.positional_encoding(height, width, self.d_model_2, device)
        K_embedded.append(temp+pe)
        src = torch.cat(K_embedded, dim=-1)
        src = src.view(batch_size, height * width, -1)
        attention_mask = torch.ones((batch_size, src.shape[1]), dtype=torch.long, device=device)
        attention_mask = attention_mask[:, None, None, :]
        output = self.transformer_encoder(inputs_embeds=src, attention_mask=attention_mask)        
        output = self.output(output)  
        
        return output.view(batch_size, height, width)


class Teqwendy_first_half(nn.Module):
    # the first half of TEQWENDY model
    # input size: batch_size * 4 * height * width
    # output size: batch_size * 4 * height * width
    def __init__(self, d_model=64, nhead=4, num_layers=7, dropout=0.1):
        super(Teqwendy_first_half, self).__init__()
        self.halfdim = d_model // 2
        self.embedding_k1 = nn.Sequential(nn.Linear(1, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.embedding_k2 = nn.Sequential(nn.Linear(1, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.embedding_k3 = nn.Sequential(nn.Linear(1, d_model), nn.ReLU(), nn.Linear(d_model, d_model))

        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=4*d_model, dropout=dropout, batch_first=True)
        self.transformer_encoder1 = nn.TransformerEncoder(encoder_layers, num_layers)
        self.transformer_encoder2 = nn.TransformerEncoder(encoder_layers, num_layers)
        self.transformer_encoder3 = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.output_k1l = nn.Sequential(nn.Linear(d_model, d_model), nn.LeakyReLU(0.1), nn.Linear(d_model, 1))
        self.output_k2l = nn.Sequential(nn.Linear(d_model, d_model), nn.LeakyReLU(0.1), nn.Linear(d_model, 1))
        self.output_k3l = nn.Sequential(nn.Linear(d_model, d_model), nn.LeakyReLU(0.1), nn.Linear(d_model, 1))
        
    
    def positional_encoding(self, height, width, device):
        
        div_term = torch.exp(
            torch.arange(0, self.halfdim, 2, device=device, dtype=torch.float) * 
            (-np.log(10000.0) / self.halfdim)
        )
        y_position = torch.arange(height, dtype=torch.float, device=device).unsqueeze(1) 
        x_position = torch.arange(width, dtype=torch.float, device=device).unsqueeze(1)  
        pe_y = torch.zeros(height, self.halfdim, device=device)
        pe_x = torch.zeros(width, self.halfdim, device=device)
        
        pe_y[:, 0::2] = torch.sin(y_position * div_term)
        pe_y[:, 1::2] = torch.cos(y_position * div_term)
        
        pe_x[:, 0::2] = torch.sin(x_position * div_term)
        pe_x[:, 1::2] = torch.cos(x_position * div_term)
        
        pe_y_expanded = pe_y.unsqueeze(1).expand(height, width, self.halfdim) 
        pe_x_expanded = pe_x.unsqueeze(0).expand(height, width, self.halfdim) 
        pe = torch.cat([pe_y_expanded, pe_x_expanded], dim=-1)
        return pe

    def forward(self, k_all):
        device = k_all.device  
        k0 = k_all[:, 0]
        k1 = k_all[:, 1]
        k2 = k_all[:, 2]
        k3 = k_all[:, 3]
        batch_size, height, width = k0.shape
        pe = self.positional_encoding(height, width, device)
        
        k1_embedded = self.embedding_k1(k1.unsqueeze(-1))
        k1_embedded += pe
        k1_embedded = k1_embedded.view(batch_size, height * width, -1).contiguous()
        output_k1l = self.transformer_encoder1(k1_embedded)
        output_k1l = self.output_k1l(output_k1l) # Shape: (batch_size, n*n, 1)
        output_k1l = output_k1l.view(batch_size, 1, height, width)
        
        k2_embedded = self.embedding_k2(k2.unsqueeze(-1))
        k2_embedded += pe
        k2_embedded = k2_embedded.view(batch_size, height * width, -1).contiguous()
        output_k2l = self.transformer_encoder2(k2_embedded)
        output_k2l = self.output_k2l(output_k2l) # Shape: (batch_size, n*n, 1)
        output_k2l = output_k2l.view(batch_size, 1, height, width)
        
        k3_embedded = self.embedding_k3(k3.unsqueeze(-1))
        k3_embedded += pe
        k3_embedded = k3_embedded.view(batch_size, height * width, -1).contiguous()
        output_k3l = self.transformer_encoder3(k3_embedded)
        output_k3l = self.output_k3l(output_k3l) # Shape: (batch_size, n*n, 1)
        output_k3l = output_k3l.view(batch_size, 1, height, width)

        output = torch.cat([k0.unsqueeze(1), output_k1l, output_k2l, output_k3l], dim=1)
        return output


class Teqwendy_second_half(nn.Module): 
    # the second half of TEQWENDY model
    # input size: batch_size * 5 * height * width
    # output size: batch_size * height * width
    def __init__(self, n=5, d_model=64, nhead=4, num_layers=3, dropout=0.1):
        super(Teqwendy_second_half, self).__init__()
        self.halfdim = d_model // 2
        self.n = n
        self.embedding = nn.ModuleList([nn.Linear(1, d_model) for _ in range(n)])
        self.segment_embedding = nn.Embedding(n, d_model)  
        encoder_layers = nn.TransformerEncoderLayer(n*d_model, nhead, dim_feedforward=4*n*d_model, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.output = nn.Linear(n*d_model, 1)

    def positional_encoding(self, height, width, device):        
        div_term = torch.exp(
            torch.arange(0, self.halfdim, 2, device=device, dtype=torch.float) * 
            (-np.log(10000.0) / self.halfdim)
        )
        y_position = torch.arange(height, dtype=torch.float, device=device).unsqueeze(1) 
        x_position = torch.arange(width, dtype=torch.float, device=device).unsqueeze(1)  
        pe_y = torch.zeros(height, self.halfdim, device=device)
        pe_x = torch.zeros(width, self.halfdim, device=device)        
        pe_y[:, 0::2] = torch.sin(y_position * div_term)
        pe_y[:, 1::2] = torch.cos(y_position * div_term)        
        pe_x[:, 0::2] = torch.sin(x_position * div_term)
        pe_x[:, 1::2] = torch.cos(x_position * div_term)        
        pe_y_expanded = pe_y.unsqueeze(1).expand(height, width, self.halfdim) 
        pe_x_expanded = pe_x.unsqueeze(0).expand(height, width, self.halfdim) 
        pe = torch.cat([pe_y_expanded, pe_x_expanded], dim=-1)
        return pe

    def forward(self, K):
        device = K.device  
        batch_size, height, width = K[:, 0].shape
        K_embedded = []
        pe = self.positional_encoding(height, width, device)
        for i in range(self.n):
            temp = self.embedding[i](K[:, i].unsqueeze(-1))
            segment_ids = i * torch.ones_like(K[:, 0], dtype=torch.long, device=device)
            se = self.segment_embedding(segment_ids)
            temp += pe + se
            K_embedded.append(temp)
        src = torch.cat(K_embedded, dim=-1)
        src = src.view(batch_size, height * width, -1)
        output = self.transformer_encoder(src)
        output = self.output(output)  
        
        return output.view(batch_size, height, width)
