import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from tqdm import tqdm
import math
import random  

# ====================
# 1. Configuration and Hyperparameters
# ====================
DATA_DIR = "./data/wn18rr"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBED_DIM = 500
BATCH_SIZE = 512
EPOCHS = 300
LEARNING_RATE = 0.0001
NEG_SAMPLES = 256
GAMMA = 6.0
PHASE_WEIGHT = 0.4
MODULUS_WEIGHT = 1.2
ADVERSARIAL_TEMP = 0.4  
WARMUP_EPOCHS = 20
REG_COEFF = 5e-5
#SEEDS = [42, 43, 44, 45, 46]  
SEEDS = [44]  

# ====================
# 1.5 Global random seed setting
# ====================
def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🌱 Random seed set to: {seed}")


# ====================
# 2. Learning rate scheduler
# ====================
class WarmupScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.epoch = 0

    def step(self):
        self.epoch += 1
        if self.epoch <= self.warmup_epochs:
            lr = self.base_lr * self.epoch / self.warmup_epochs
        else:
            progress = (self.epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.base_lr * (0.5 * (1.0 + math.cos(math.pi * progress)))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

# ====================
# 3. Unified data set loader
# ====================
class UnifiedKGDataset(Dataset):
    def __init__(self, path, entity2id, relation2id):
        self.data = []
        print(f"Loading data: {path}...")
        with open(path, 'r') as f:
            for line in f:
                h, r, t = line.strip().split('\t')
                if h in entity2id and r in relation2id and t in entity2id:
                    self.data.append((entity2id[h], relation2id[r], entity2id[t]))
        print(f"Loading completed: {len(self.data)} number of triples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx])


def load_unified_datasets(data_dir):
    train_path = os.path.join(data_dir, "train.txt")
    valid_path = os.path.join(data_dir, "valid.txt")
    test_path = os.path.join(data_dir, "test.txt")

    entity2id = {}
    relation2id = {}
    print("Build a unified entity and relationship mapping table...")
    with open(train_path, 'r') as f:
        for line in f:
            h, r, t = line.strip().split('\t')
            if h not in entity2id: entity2id[h] = len(entity2id)
            if t not in entity2id: entity2id[t] = len(entity2id)
            if r not in relation2id: relation2id[r] = len(relation2id)

    num_entities = len(entity2id)
    num_relations = len(relation2id)
    print(f"Number of entities: {num_entities}, Number of relations: {num_relations}")

    train_dataset = UnifiedKGDataset(train_path, entity2id, relation2id)
    valid_dataset = UnifiedKGDataset(valid_path, entity2id, relation2id)
    test_dataset = UnifiedKGDataset(test_path, entity2id, relation2id)

    return train_dataset, valid_dataset, test_dataset, num_entities, num_relations

# ====================
# 4. HKG Module
# ====================
class AdvancedHKG(nn.Module):
    def __init__(self, num_ent, num_rel, hidden_dim, gamma=GAMMA,
                 modulus_weight=MODULUS_WEIGHT, phase_weight=PHASE_WEIGHT):
        super(AdvancedHKG, self).__init__()
        self.num_entity = num_ent
        self.num_relation = num_rel
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0

        self.gamma = nn.Parameter(torch.Tensor([gamma]), requires_grad=False)
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        self.entity_embedding = nn.Parameter(torch.zeros(num_ent, hidden_dim * 3))
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.relation_embedding = nn.Parameter(torch.zeros(num_rel, hidden_dim * 3))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        #self.phase_weight = nn.Parameter(torch.Tensor([[phase_weight * self.embedding_range.item()]]))
        self.phase_weight = nn.Parameter(torch.Tensor([[phase_weight]]))  # 直接使用 0.4
        self.modulus_weight = nn.Parameter(torch.Tensor([[modulus_weight]]))
        self.pi = 3.14159265358979323846

    def func(self, head, rel, tail):
        mod_head, phase1_head, phase2_head = torch.chunk(head, 3, dim=-1)
        mod_rel, phase1_rel, phase2_rel = torch.chunk(rel, 3, dim=-1)
        mod_tail, phase1_tail, phase2_tail = torch.chunk(tail, 3, dim=-1)

        scale = self.embedding_range.item() / self.pi
        phase1_head = phase1_head / scale
        phase2_head = phase2_head / scale
        phase1_rel = phase1_rel / scale
        phase2_rel = phase2_rel / scale
        phase1_tail = phase1_tail / scale
        phase2_tail = phase2_tail / scale

        mod_diff = mod_head * mod_rel - mod_tail

        phase1_diff = phase1_head + phase1_rel - phase1_tail
        phase2_diff = phase2_head + phase2_rel - phase2_tail

        modulus_term = torch.sqrt(torch.sum(mod_diff ** 2, dim=-1) + 1e-8) * self.modulus_weight
        phase1_term = torch.sqrt(torch.sum(torch.sin(phase1_diff / 2) ** 2, dim=-1) + 1e-8) * self.phase_weight
        phase2_term = torch.sqrt(torch.sum(torch.sin(phase2_diff / 2) ** 2, dim=-1) + 1e-8) * self.phase_weight

        total_score = modulus_term + phase1_term + phase2_term
        return self.gamma.item() - total_score

    def forward(self, h_idx, r_idx, t_idx):
        head = torch.index_select(self.entity_embedding, dim=0, index=h_idx).unsqueeze(1)
        rel = torch.index_select(self.relation_embedding, dim=0, index=r_idx).unsqueeze(1)
        tail = torch.index_select(self.entity_embedding, dim=0, index=t_idx).unsqueeze(1)
        return self.func(head, rel, tail).squeeze(1)

    def score_tail(self, head_idx, rel_idx, tail_candidates_idx):
        """Tail entity prediction (h, r, ?)"""
        head = torch.index_select(self.entity_embedding, dim=0, index=head_idx).unsqueeze(1)
        rel = torch.index_select(self.relation_embedding, dim=0, index=rel_idx).unsqueeze(1)

        if len(tail_candidates_idx.shape) == 1:
            B = tail_candidates_idx.size(0)
            tail = torch.index_select(self.entity_embedding, dim=0, index=tail_candidates_idx).unsqueeze(1)
            head_expanded = head.expand(B, -1, -1)
            rel_expanded = rel.expand(B, -1, -1)
            return self.func(head_expanded, rel_expanded, tail).squeeze(-1)
        else:
            B, N = tail_candidates_idx.shape
            tail = torch.index_select(self.entity_embedding, dim=0,
                                      index=tail_candidates_idx.view(-1)).view(B, N, -1)
            head_expanded = head.expand(B, N, -1)
            rel_expanded = rel.expand(B, N, -1)
            return self.func(head_expanded, rel_expanded, tail)

    def score_head(self, rel_idx, tail_idx, head_candidates_idx):
        """Head entity prediction (?, r, t)"""
        rel = torch.index_select(self.relation_embedding, dim=0, index=rel_idx).unsqueeze(1)
        tail = torch.index_select(self.entity_embedding, dim=0, index=tail_idx).unsqueeze(1)

        if len(head_candidates_idx.shape) == 1:
            B = head_candidates_idx.size(0)
            head = torch.index_select(self.entity_embedding, dim=0, index=head_candidates_idx).unsqueeze(1)
            rel_expanded = rel.expand(B, -1, -1)
            tail_expanded = tail.expand(B, -1, -1)
            return self.func(head, rel_expanded, tail_expanded).squeeze(-1)
        else:
            B, N = head_candidates_idx.shape
            head = torch.index_select(self.entity_embedding, dim=0,
                                      index=head_candidates_idx.view(-1)).view(B, N, -1)
            rel_expanded = rel.expand(B, N, -1)
            tail_expanded = tail.expand(B, N, -1)
            return self.func(head, rel_expanded, tail_expanded)

# ====================
# 5. Training steps
# ====================
def train_step_paper_config(model, dataloader, optimizer, scheduler, num_ent, epoch):
    model.train()
    total_loss = 0
    batch_count = 0

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training", leave=False)):
        h, r, t = batch[:, 0].to(DEVICE), batch[:, 1].to(DEVICE), batch[:, 2].to(DEVICE)

        # Positive sample score
        pos_score = model(h, r, t)
        batch_size = h.size(0)

        # Negative tail sample
        neg_t = torch.randint(0, num_ent, (batch_size, NEG_SAMPLES // 2), device=DEVICE)
        neg_t_scores = model.score_tail(
            h.repeat_interleave(NEG_SAMPLES // 2),
            r.repeat_interleave(NEG_SAMPLES // 2),
            neg_t.view(-1)
        ).view(batch_size, NEG_SAMPLES // 2)

        # Negative sample
        neg_h = torch.randint(0, num_ent, (batch_size, NEG_SAMPLES // 2), device=DEVICE)
        neg_h_scores = model.score_head(
            r.repeat_interleave(NEG_SAMPLES // 2),
            t.repeat_interleave(NEG_SAMPLES // 2),
            neg_h.view(-1)
        ).view(batch_size, NEG_SAMPLES // 2)

        # Merge negative samples
        neg_scores = torch.cat([neg_t_scores, neg_h_scores], dim=1)
        neg_weights = F.softmax(neg_scores * ADVERSARIAL_TEMP, dim=1).detach()

        # Loss calculation
        pos_loss = F.logsigmoid(pos_score).squeeze()
        neg_loss = torch.sum(neg_weights * F.logsigmoid(-neg_scores), dim=1)
        loss = -(torch.mean(pos_loss) + torch.mean(neg_loss)) / 2

        # L2 regularization
        entity_reg = torch.mean(torch.norm(model.entity_embedding, p=2, dim=1) ** 2)
        #relation_reg = torch.mean(torch.norm(model.relation_embedding, p=2, dim=1) ** 2)
        #loss = loss + REG_COEFF * (entity_reg + relation_reg)
        loss = loss + REG_COEFF * entity_reg

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        batch_count += 1

    scheduler.step()
    return total_loss / batch_count

# ====================
# 6. ✅ Bidirectional complete filtering assessment
# ====================
@torch.no_grad()
def evaluate_filtered_full(model, dataset, train_dataset, valid_dataset, test_dataset,
                           num_entities, mode='Test'):
    """Bidirectional filtering assessment (Head Prediction + Tail Prediction)"""
    model.eval()

    hits_at_1 = 0; hits_at_3 = 0; hits_at_10 = 0
    mrr = 0.0; mr = 0.0; count = 0

    # Separately count the head/tail indicators for diagnosis
    mrr_tail_sum = 0.0; mrr_head_sum = 0.0
    count_tail = 0; count_head = 0

    all_triples_set = set(train_dataset.data + valid_dataset.data + test_dataset.data)
    all_entities = torch.arange(num_entities, device=DEVICE)

    # ✅ Preprocessing
    all_triples_tensor = torch.tensor(
        list(all_triples_set), dtype=torch.long, device=DEVICE
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    print(f"\nStart {mode} Bidirectional complete filtering assessment...")
    for batch in tqdm(loader, desc=f"Evaluating {mode} (Bidirectional Filtered)"):
        h, r, t = batch[:, 0].to(DEVICE), batch[:, 1].to(DEVICE), batch[:, 2].to(DEVICE)
        current_h = h.item(); current_r = r.item(); current_t = t.item()

        # ===== 1. Tail entity prediction (h, r, ?) =====
        scores_tail = model.score_tail(
            h.repeat(num_entities), r.repeat(num_entities), all_entities
        ).squeeze()

        # ✅ Vectorized filtering
        mask_tail = (all_triples_tensor[:, 0] == current_h) & \
                    (all_triples_tensor[:, 1] == current_r)
        false_negatives_tail = all_triples_tensor[mask_tail, 2]

        filtered_scores_tail = scores_tail.clone()
        filtered_scores_tail[false_negatives_tail] = float('-inf')
        filtered_scores_tail[current_t] = scores_tail[current_t]

        _, sorted_indices_tail = torch.sort(filtered_scores_tail, descending=True)
        true_rank_tail = (sorted_indices_tail == current_t).nonzero(as_tuple=True)[0].item() + 1

        # ===== 2. Head entity prediction (?, r, t) =====
        scores_head = model.score_head(
            r.repeat(num_entities), t.repeat(num_entities), all_entities
        ).squeeze()

        # ✅ Vectorized filtering
        mask_head = (all_triples_tensor[:, 1] == current_r) & \
                    (all_triples_tensor[:, 2] == current_t)
        false_negatives_head = all_triples_tensor[mask_head, 0]

        filtered_scores_head = scores_head.clone()
        filtered_scores_head[false_negatives_head] = float('-inf')
        filtered_scores_head[current_h] = scores_head[current_h]

        _, sorted_indices_head = torch.sort(filtered_scores_head, descending=True)
        true_rank_head = (sorted_indices_head == current_h).nonzero(as_tuple=True)[0].item() + 1

        # ===== 3. Cumulative bidirectional indicator =====
        for rank in [true_rank_tail, true_rank_head]:
            mr += rank; mrr += 1.0 / rank; count += 1
            if rank <= 1: hits_at_1 += 1
            if rank <= 3: hits_at_3 += 1
            if rank <= 10: hits_at_10 += 1

        # Accumulate separately from the beginning to the end MRR
        mrr_tail_sum += 1.0 / true_rank_tail; count_tail += 1
        mrr_head_sum += 1.0 / true_rank_head; count_head += 1

    mr /= count; mrr /= count
    hits_at_1 /= count; hits_at_3 /= count; hits_at_10 /= count

    mrr_tail = mrr_tail_sum / max(count_tail, 1)
    mrr_head = mrr_head_sum / max(count_head, 1)

    print(f"--- {mode} Bidirectional Full Filtered Results ---")
    print(f"MR: {mr:.4f}")
    print(f"MRR: {mrr:.4f}  (Tail: {mrr_tail:.4f} | Head: {mrr_head:.4f} | Gap: {abs(mrr_tail-mrr_head):.4f})")
    print(f"Hits@1: {hits_at_1:.4f}")
    print(f"Hits@3: {hits_at_3:.4f}")
    print(f"Hits@10: {hits_at_10:.4f}")

    return mrr

# ====================
# 7. Main program
# ====================
def run_single_seed(seed, train_dataset, valid_dataset, test_dataset, num_ent, num_rel):
    set_global_seed(seed)

    model = AdvancedHKG(num_ent, num_rel, EMBED_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-6)
    scheduler = WarmupScheduler(optimizer, WARMUP_EPOCHS, EPOCHS, LEARNING_RATE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    best_mrr = 0.0
    best_epoch = 0
    best_state = None
    save_name = f"best_hkg_model_seed{seed}.pt"

    for epoch in range(EPOCHS):
        loss = train_step_paper_config(model, train_loader, optimizer, scheduler, num_ent, epoch)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Epoch [{epoch+1}/{EPOCHS}], Loss: {loss:.4f}, LR: {current_lr:.6f}")

        if (epoch + 1) % 20 == 0:
            val_mrr = evaluate_filtered_full(
                model, valid_dataset,
                train_dataset, valid_dataset, test_dataset,
                num_ent, mode='Validation'
            )
            if val_mrr > best_mrr:
                best_mrr = val_mrr
                best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                print(f"  🏆 New best MRR: {best_mrr:.4f} at epoch {epoch+1}")

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(DEVICE)

    test_mrr = evaluate_filtered_full(
        model, test_dataset,
        train_dataset, valid_dataset, test_dataset,
        num_ent, mode=f'Test (Seed {seed})'
    )

    torch.save({
        'seed': seed, 'epoch': best_epoch, 'mrr': test_mrr,
        'model_state_dict': best_state if best_state else model.state_dict(),
    }, save_name)

    return {"seed": seed, "MRR": test_mrr, "best_epoch": best_epoch + 1}


def main():
    print("Starting HKG multi-seed training...")

    train_dataset, valid_dataset, test_dataset, num_ent, num_rel = load_unified_datasets(DATA_DIR)
    print(f"Entities: {num_ent}, Relations: {num_rel}")
    print(f"Train: {len(train_dataset)}, Valid: {len(valid_dataset)}, Test: {len(test_dataset)}")

    all_results = []
    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"🌱 Seed: {seed} | Device: {DEVICE}")
        print(f"{'='*60}")

        result = run_single_seed(seed, train_dataset, valid_dataset, test_dataset, num_ent, num_rel)
        all_results.append(result)
        print(f"  📊 Seed {seed} Test MRR: {result['MRR']:.4f}")

    # ✅ Summary statistics Mean ± Std
    print(f"\n{'='*60}")
    print("📈 Final Multi-Seed Results (Mean ± Std)")
    print(f"{'='*60}")
    mrr_values = [r["MRR"] for r in all_results]
    print(f"  MRR: {np.mean(mrr_values):.4f} ± {np.std(mrr_values):.4f}")
    print(f"  Individual: {[f'{v:.4f}' for v in mrr_values]}")

    # Save the summary results
    np.save('multi_seed_results.npy', np.array(all_results))
    print("✅ The summary results have been saved to 'multi_seed_results.npy'")


if __name__ == "__main__":
    main()