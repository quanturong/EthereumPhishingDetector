# EthereumPhishingDetector

Đánh giá và tăng cường độ bền vững của mô hình phát hiện lừa đảo trên Ethereum trước các tấn công né tránh dựa trên đồ thị giao dịch.

**Khóa luận tốt nghiệp** – Đại học Công nghệ Thông tin, ĐHQG-HCM
**Cán bộ hướng dẫn**: TS. Phan Thế Duy

## Tổng quan

Repository chứa toàn bộ mã nguồn cho pipeline nghiên cứu:

1. **Xây dựng dataset EthPhishGraph-2026** (~7,036 target accounts, snapshot 09/2026)
2. **Ba baseline GNN**: GraphSAGE, GATv2, PEAE-style GNN
3. **Sáu chiến lược tấn công đối kháng ở mức giao dịch**:
   - Baseline (Hide and Seek): TOH, TIH, BPH-T2P, BPH-P2T
   - Novel (đề xuất): STC (Split-and-Temporal Camouflage), MWR (Multi-Wallet Routing)
4. **Hai phương pháp phòng thủ**: Adversarial Training + Reliability-Aware GNN
5. **Cross-check** trên dataset D2 (XBLOCK)

## Cấu trúc pipeline

| Stage | Script | Mô tả |
|-------|--------|-------|
| 1 | – | Load 3,518 phishing addresses từ forta-network |
| 2 | `crawl_normal_candidates.py` | Crawl hop-1 + stratified sample 3,518 normal candidates |
| 3 | `stage3_pipeline.py` | Temporal split 60/20/20 + build ego-graph K=1 |
| 4 | `stage4_imbalanced_testonly.py` | Test-only pool cho kịch bản mất cân bằng 1:5, 1:10 |
| 5 | `stage5_train_baselines.py` | Train 3 baseline models × 3 seeds |
| 6 | `stage6_adversarial_attacks.py` | Đánh giá 6 attacks × 5 budgets |
| 7 | `stage7_defense.py` | AdvTrain + RA-SAGE defense |
| 8 | `stage8_crosscheck_D2.py` | Cross-check trên XBLOCK D2 |
| 9 | `stage9_k2_upgrade.py` | Upgrade ego-graph K=1 → K=2 |
| – | `prepare_dataset_bundle.py` | Bundle dataset để chia sẻ |
| – | `host_dataset.py` | HTTP server host dataset bundle |

## Yêu cầu

```
python >= 3.10
torch >= 2.0
torch_geometric >= 2.8
pandas, numpy, scikit-learn, requests, tqdm
```

## Sử dụng nhanh

### 1. Chuẩn bị dữ liệu

```bash
# Crawl phishing hop-1 txs (cần Etherscan V2 API key)
python crawl_normal_candidates.py --api_key YOUR_KEY --step 2a

# Build normal candidates
python crawl_normal_candidates.py --api_key YOUR_KEY --step 2b
python crawl_normal_candidates.py --api_key YOUR_KEY --step 2c --pool_sample_size 30000
python crawl_normal_candidates.py --api_key YOUR_KEY --step 2d

# Build ego-graph K=1
python stage3_pipeline.py --api_key YOUR_KEY --step all

# (Optional) Upgrade lên K=2
python stage9_k2_upgrade.py --step 9a
python stage9_k2_upgrade.py --api_key YOUR_KEY --step 9b   # ~9 giờ crawl
python stage9_k2_upgrade.py --step 9c
```

### 2. Train baselines

```bash
python stage5_train_baselines.py --model all --seeds 42 43 44 --epochs 100
```

### 3. Chạy adversarial attacks

```bash
python stage6_adversarial_attacks.py --model graphsage --budgets 1 3 5 10 20
python stage6_adversarial_attacks.py --model gatv2     --budgets 1 3 5 10 20
python stage6_adversarial_attacks.py --model peaegnn   --budgets 1 3 5 10 20
```

### 4. Đánh giá phòng thủ

```bash
python stage7_defense.py --method both --seeds 42 43 44 --epochs 60
```

## Dataset

Dataset EthPhishGraph-2026 quá lớn để đưa lên GitHub. Được host riêng qua HTTP hoặc release riêng.

## Tài liệu tham khảo chính

- [1] Huang et al., "PEAE-GNN: Phishing Detection on Ethereum via Augmentation Ego-Graph", IEEE TCSS 2024
- [2] Wen et al., "Hide and Seek: An Adversarial Hiding Approach Against Phishing Detection on Ethereum", IEEE TCSS 2023
- [13] Hamilton et al., "Inductive Representation Learning on Large Graphs" (GraphSAGE), NeurIPS 2017
- [14] Brody et al., "How Attentive are Graph Attention Networks?" (GATv2), ICLR 2022
- [15] Yuan et al., "XBLOCK Blockchain Datasets: InPlusLab Ethereum Second-order Phishing Datasets", 2020
