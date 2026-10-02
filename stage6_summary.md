## Hieu nang sach (clean) tren graph dung lai - trung binh ± do lech chuan qua seed
| Mo hinh | So seed | Macro-F1 | Phishing recall | So mau phishing dung (TB) |
|---|---|---|---|---|
| peaegnn | 3 | 88.13 ± 0.28 | 82.8 ± 0.6 % | 492/594 |
| graphsage | 3 | 86.63 ± 0.32 | 77.7 ± 1.1 % | 462/594 |
| gatv2 | 3 | 81.51 ± 1.08 | 66.2 ± 1.6 % | 393/594 |

## ASR (%) - peaegnn - trung binh ± do lech chuan qua 3 seed
| Tan cong | nhom | b=1 | b=3 | b=5 | b=10 | b=20 | TB cac budget |
|---|---|---|---|---|---|---|---|
| TOH | known | 0.4 ± 0.2 | 1.2 ± 0.4 | 1.2 ± 0.3 | 1.8 ± 0.6 | 1.9 ± 1.1 | 1.3 ± 0.5 |
| TIH | known | 0.5 ± 0.4 | 0.5 ± 0.3 | 0.5 ± 0.4 | 0.5 ± 0.4 | 0.8 ± 0.8 | 0.6 ± 0.4 |
| BPH-T2P | known | 0.1 ± 0.1 | 0.2 ± 0.0 | 0.3 ± 0.1 | 0.8 ± 0.3 | 1.3 ± 0.7 | 0.5 ± 0.2 |
| BPH-P2T | known | 1.1 ± 0.8 | 1.2 ± 0.8 | 1.4 ± 0.8 | 1.5 ± 1.0 | 1.4 ± 1.2 | 1.3 ± 0.9 |
| STC | unseen | 1.3 ± 0.4 | 2.1 ± 0.7 | 2.3 ± 0.6 | 3.1 ± 0.9 | 3.4 ± 1.1 | 2.4 ± 0.7 |
| MWR | unseen | 0.1 ± 0.1 | 0.1 ± 0.1 | 0.1 ± 0.1 | 0.1 ± 0.2 | 0.1 ± 0.2 | 0.1 ± 0.1 |

ASR_any (lat boi it nhat 1 tan cong/ngan sach): 5.3 ± 2.5 %  (so mau lat TB: 26.0)

## ASR (%) - graphsage - trung binh ± do lech chuan qua 3 seed
| Tan cong | nhom | b=1 | b=3 | b=5 | b=10 | b=20 | TB cac budget |
|---|---|---|---|---|---|---|---|
| TOH | known | 1.3 ± 0.6 | 1.2 ± 0.5 | 1.2 ± 0.7 | 1.2 ± 0.5 | 1.5 ± 0.3 | 1.3 ± 0.5 |
| TIH | known | 0.6 ± 0.4 | 0.6 ± 0.2 | 1.1 ± 0.3 | 1.2 ± 0.6 | 1.6 ± 0.9 | 1.0 ± 0.4 |
| BPH-T2P | known | 0.7 ± 0.3 | 0.9 ± 0.5 | 0.9 ± 0.3 | 1.2 ± 0.4 | 1.8 ± 0.5 | 1.1 ± 0.2 |
| BPH-P2T | known | 2.6 ± 2.3 | 2.7 ± 2.3 | 2.8 ± 2.6 | 3.1 ± 2.7 | 3.2 ± 3.1 | 2.9 ± 2.6 |
| STC | unseen | 1.7 ± 0.7 | 2.5 ± 1.6 | 2.7 ± 1.5 | 3.2 ± 1.6 | 3.5 ± 1.9 | 2.7 ± 1.5 |
| MWR | unseen | 0.5 ± 0.1 | 0.6 ± 0.1 | 0.6 ± 0.3 | 0.5 ± 0.3 | 0.7 ± 0.6 | 0.6 ± 0.2 |

ASR_any (lat boi it nhat 1 tan cong/ngan sach): 8.7 ± 4.9 %  (so mau lat TB: 40.0)

## ASR (%) - gatv2 - trung binh ± do lech chuan qua 3 seed
| Tan cong | nhom | b=1 | b=3 | b=5 | b=10 | b=20 | TB cac budget |
|---|---|---|---|---|---|---|---|
| TOH | known | 1.1 ± 0.5 | 1.9 ± 0.7 | 2.8 ± 0.4 | 3.7 ± 0.7 | 4.6 ± 1.2 | 2.8 ± 0.6 |
| TIH | known | 0.0 ± 0.0 | 0.1 ± 0.1 | 0.2 ± 0.1 | 0.9 ± 0.1 | 1.5 ± 0.2 | 0.5 ± 0.0 |
| BPH-T2P | known | 0.5 ± 0.4 | 0.8 ± 0.0 | 1.5 ± 0.2 | 2.7 ± 0.2 | 4.0 ± 0.4 | 1.9 ± 0.1 |
| BPH-P2T | known | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.1 ± 0.1 | 0.5 ± 0.0 | 0.1 ± 0.0 |
| STC | unseen | 1.4 ± 0.5 | 2.0 ± 0.3 | 2.2 ± 0.5 | 2.5 ± 0.7 | 3.2 ± 1.1 | 2.2 ± 0.6 |
| MWR | unseen | 0.1 ± 0.1 | 0.1 ± 0.1 | 0.1 ± 0.1 | 0.3 ± 0.3 | 0.4 ± 0.4 | 0.2 ± 0.2 |

ASR_any (lat boi it nhat 1 tan cong/ngan sach): 6.2 ± 1.0 %  (so mau lat TB: 24.3)

## Huong tac dong len P(phishing) - budget lon nhat (thay doi so voi clean, don vi diem xac suat)
| Mo hinh | TOH | TIH | BPH-T2P | BPH-P2T | STC | MWR |
|---|---|---|---|---|---|---|
| peaegnn | -0.031 ± 0.005 | +0.055 ± 0.006 | -0.018 ± 0.010 | +0.074 ± 0.005 | -0.069 ± 0.006 | +0.057 ± 0.006 |
| graphsage | -0.012 ± 0.002 | +0.009 ± 0.002 | +0.015 ± 0.014 | +0.036 ± 0.004 | -0.037 ± 0.017 | +0.056 ± 0.024 |
| gatv2 | -0.045 ± 0.004 | +0.013 ± 0.004 | -0.034 ± 0.003 | +0.040 ± 0.004 | -0.031 ± 0.006 | +0.007 ± 0.011 |

(am = tan cong lam mo hinh bot tin la phishing; duong = tan cong vo tinh giup mo hinh nhan ra phishing hon)

## Kiem tra du lieu
- peaegnn: 3 seed | giao dich khong hop le: 0 | target bi bo qua: 0
- graphsage: 3 seed | giao dich khong hop le: 0 | target bi bo qua: 0
- gatv2: 3 seed | giao dich khong hop le: 0 | target bi bo qua: 0