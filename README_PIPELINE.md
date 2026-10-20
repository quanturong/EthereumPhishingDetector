# Đánh giá độ bền của mô hình phát hiện phishing Ethereum trước tấn công trên đồ thị giao dịch

Mã nguồn của khóa luận (UIT). Dữ liệu: EthPhishGraph-2026 (ego-graph K=2, 7.036 ví: 3.518 phishing và 3.518 normal), chia theo thời gian 4.221 / 1.407 / 1.408.
Bản dữ liệu trên Hugging Face (https://huggingface.co/datasets/quanturong/EthPhishGraph-2026) **đang được đối chiếu** với bản dùng để chạy thí nghiệm (số cạnh trung bình trên thẻ dataset chưa khớp log huấn luyện).

## Mô hình
- **`peae_paper` (mô hình chính):** PEAE-GNN bản bám sát bài Huang et al., IEEE TCSS 2024 (đặc trưng tăng cường, kiểu GraphSAGE có self-loop, readout RTM top-3). Đặc trưng tính từ danh sách cạnh nên dùng thống kê riêng.
- `graphsage`, `gatv2`: mô hình so sánh.
- `peaegnn` (PEAE-style, xấp xỉ), `peaegnnv2`: mô hình cũ, chỉ dùng làm phụ lục; **không khớp bài gốc**.

## Tệp chính và thứ tự chạy
| Bước | Tệp | Ghi chú |
|---|---|---|
| Dữ liệu | `stage9_k2_upgrade.py` | dựng ego-graph K=2 |
| Stage 5: huấn luyện | `stage5_train_baselines.py` | `--model peae_paper` cần `--out` thư mục riêng; `--model all` không gồm `peae_paper` |
| Stage 6: tấn công | `stage6_adversarial_attacks.py`, `run_mincost_all.py` | 7 tấn công; chế độ chi phí tối thiểu |
| Độ nhạy đặc trưng | `diag_peae_features.py` (peae_paper), `diag_feature_sensitivity.py` (mô hình khác) | thay đặc trưng trực tiếp, không phải tấn công thật |
| Stage 7: phòng thủ | `stage7_defense.py` (train trên GPU, eval trên máy) | `advtrain`, `ra` (độ tin cậy cạnh), `ra_noedge` |
| Tổng hợp | `summarize_stage5.py`, `summarize_mincost.py`, `summarize_diag.py`, `summarize_stage7.py`, `analyze_stage7_auc.py`, `stats_stage7.py` | |
| Khác | `eval_stage5_rocauc.py`, `collect_json_stage5_6.py`, `package_results.py`, `make_stage7_bundle.py`, `stage8_eval_d2.py` | |

Ví dụ:
```
python stage5_train_baselines.py --model peae_paper --bs 32 --epochs 64 --seeds 42 43 44 --pkl <ego_graph_k2_dataset.pkl> --out <stage5_peae>
python run_mincost_all.py --models peae_paper --seeds 42 43 44 --tag_suffix full
python summarize_mincost.py --models peae_paper graphsage gatv2 --seeds 42 43 44 --tag_suffix full
```
Đường dẫn mặc định trong mã là đường dẫn Windows của người viết (`D:\NCKH\Final\...`); đổi bằng tham số dòng lệnh.

## Tình trạng
Stage 5 và 6 xong cho `peae_paper`, GraphSAGE, GATv2 (3 seed). Stage 7 cho `peae_paper` đang huấn luyện; D2 chưa chạy cho `peae_paper`. Nhật ký thí nghiệm: `docs/NhatKy_ThucNghiem_Stage5-8.md`.

## Giới hạn chính
3 seed; tấn công cố định (không thích nghi theo mô hình); `peae_paper` đạt macro-F1 0,834, thấp hơn các xấp xỉ; mọi giả thuyết về cơ chế trong nhật ký chưa được kiểm chứng.
