import os
import torch
from tqdm import tqdm
from typing import Optional
import math

from need_to_do.transformer import MoETransformer
from config import ModelConfig, TrainConfig

class Trainer():
    def __init__(self, train_cfg: TrainConfig, model_cfg: ModelConfig, train_dataloader, val_dataloader):
        
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        self.model = MoETransformer(self.model_cfg)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), train_cfg.lr)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model.to(self.device)
        
    def train_step(self):
        self.model.train()
        
        total_loss = 0.0
        total_aux_loss = 0.0
        for batch in tqdm(self.train_dataloader, desc="Training step..."):
            
            self.optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(self.device)
            attn_mask = batch["attention_mask"].to(self.device)
            labels = batch.get("labels", input_ids).to(self.device)
            
            logits, aux_loss = self.model(input_ids, attn_mask) # ((bs, seq_len, vocab_size), scalar)
            
            logits = logits[:,:-1,:].contiguous()
            targets = input_ids[:,1:].contiguous()
            
            loss = self.loss_fn(
                logits.view(-1, self.model_cfg.vocab_size),
                targets.view(-1)
            )
            loss += self.model_cfg.load_balancing_weight * aux_loss
            
            total_loss += loss.item()
            total_aux_loss += aux_loss.item()
            
            loss.backward()
            self.optimizer.step()
        
        avg_ce_loss = total_loss / len(self.train_dataloader)
        avg_aux_loss = total_aux_loss / len(self.train_dataloader)
        
        perplexity = math.exp(avg_ce_loss)
        
        return avg_ce_loss, avg_aux_loss, perplexity
            
    def validate_step(self):
        self.model.eval()
        
        total_loss = 0.0
        total_aux_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Evaluating step..."):
                input_ids = batch["input_ids"].to(self.device)
                attn_mask = batch["attention_mask"].to(self.device)
                labels = batch.get("labels", input_ids).to(self.device)
                
                logits, aux_loss = self.model(input_ids, attn_mask) # ((bs, seq_len, vocab_size), scalar)
                
                logits = logits[:,:-1,:].contiguous()
                targets = input_ids[:,1:].contiguous()
                
                loss = self.loss_fn(
                    logits.view(-1, self.model_cfg.vocab_size),
                    targets.view(-1)
                )
                
                total_loss += loss.item()
                total_aux_loss += aux_loss.item()
                
            avg_ce_loss = total_loss / len(self.val_dataloader)
            avg_aux_loss = total_aux_loss / len(self.val_dataloader)
            
            perplexity = math.exp(avg_ce_loss)
            
        return avg_ce_loss, avg_aux_loss, perplexity
            
    def train(self):
        
        for epoch in tqdm(range(self.train_cfg.epochs), desc="Training..."):
            print(f"{epoch + 1} Epoch" + "="*50 + "\n")
            train_avg_ce_loss, train_avg_aux_loss, train_perpelxity = self.train_step()
            print(f"Train loss: {train_avg_ce_loss:.2f} | Train aux loss: {train_avg_aux_loss:.2f} | Train ppx: {train_perpelxity:.2f}\n")
            
            val_avg_ce_loss, val_avg_aux_loss, val_perpelxity = self.validate_step()
            print(f"Val loss: {val_avg_ce_loss:.2f} | Val aux loss: {val_avg_aux_loss:.2f} | Val ppx: {val_perpelxity:.2f}\n\n")
    
    def save(self, epoch: int, file_name: Optional[str] = None) -> None:
        
        save_dir = self.train_cfg.save_model_dir
        
        os.makedirs(save_dir, exist_ok=True)
        
        if file_name is None:
            save_path = os.path.join(save_dir, f"checkpoint_{epoch + 1}.pth")
        else:
            save_path = os.path.join(save_dir, file_name)
        
        try:
            checkpoint_dict = {
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }
            torch.save(
                checkpoint_dict,
                save_path
            )
            print(f"Checkpoint successfully saved to {save_path}")
        except Exception as e:
            print(f"Error while saving model:\n{e}")
    
    def load_checkpoint(self, load_path: str):
        
        try:
            checkpoint = torch.load(load_path)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            print(f"Sucessfully loaded model from {start_epoch} epoch")
        except Exception as e:
            print(f"Error while loading model\n{e}")
            
        return start_epoch, self.model, self.optimizer