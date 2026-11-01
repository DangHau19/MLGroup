# BÀN GIAO EDA SV2 - TUẦN 02

## Thông tin
- Người thực hiện: Phan Lê Minh Huy (SV2)
- Issue: #3
- Dataset: UCI 697
- Số mẫu: 4424
- Số đặc trưng: 36
- Giá trị thiếu: 0
- Dòng trùng lặp: 0

## Danh sách nhóm biến

### Categorical (10 biến)
- `Marital Status`
- `Application mode`
- `Course`
- `Daytime/evening attendance`
- `Previous qualification`
- `Nationality`
- `Mother's qualification`
- `Father's qualification`
- `Mother's occupation`
- `Father's occupation`

### Binary (7 biến)
- `Displaced`
- `Educational special needs`
- `Debtor`
- `Tuition fees up to date`
- `Gender`
- `Scholarship holder`
- `International`

### Numeric (7 biến)
- `Application order`
- `Previous qualification (grade)`
- `Admission grade`
- `Age at enrollment`
- `Unemployment rate`
- `Inflation rate`
- `GDP`

### Curricular 1st (6 biến)
- `Curricular units 1st sem (credited)`
- `Curricular units 1st sem (enrolled)`
- `Curricular units 1st sem (evaluations)`
- `Curricular units 1st sem (approved)`
- `Curricular units 1st sem (grade)`
- `Curricular units 1st sem (without evaluations)`

### Curricular 2nd (6 biến)
- `Curricular units 2nd sem (credited)`
- `Curricular units 2nd sem (enrolled)`
- `Curricular units 2nd sem (evaluations)`
- `Curricular units 2nd sem (approved)`
- `Curricular units 2nd sem (grade)`
- `Curricular units 2nd sem (without evaluations)`

## Các cặp biến tương quan cao

| Biến 1 | Biến 2 | Pearson r |
|---|---|---:|
| Curricular units 1st sem (credited) | Curricular units 2nd sem (credited) | 0.9448 |
| Curricular units 1st sem (enrolled) | Curricular units 2nd sem (enrolled) | 0.9426 |
| Curricular units 1st sem (approved) | Curricular units 2nd sem (approved) | 0.9040 |
| Curricular units 1st sem (grade) | Curricular units 2nd sem (grade) | 0.8372 |
| Curricular units 1st sem (evaluations) | Curricular units 2nd sem (evaluations) | 0.7789 |
| Curricular units 1st sem (credited) | Curricular units 1st sem (enrolled) | 0.7743 |
| Curricular units 1st sem (enrolled) | Curricular units 1st sem (approved) | 0.7691 |
| Curricular units 2nd sem (approved) | Curricular units 2nd sem (grade) | 0.7608 |
| Curricular units 1st sem (enrolled) | Curricular units 2nd sem (credited) | 0.7537 |
| Curricular units 1st sem (approved) | Curricular units 2nd sem (enrolled) | 0.7338 |
| Curricular units 2nd sem (enrolled) | Curricular units 2nd sem (approved) | 0.7033 |
| Curricular units 1st sem (approved) | Curricular units 1st sem (grade) | 0.6961 |
| Curricular units 1st sem (approved) | Curricular units 2nd sem (grade) | 0.6856 |
| Curricular units 1st sem (enrolled) | Curricular units 1st sem (evaluations) | 0.6802 |
| Curricular units 2nd sem (credited) | Curricular units 2nd sem (enrolled) | 0.6763 |

## Mã phân loại hiếm (dưới 20 mẫu)

| Biến | Mã | Số lượng |
|---|---|---:|
| Application mode | 57 | 1 |
| Application mode | 26 | 1 |
| Application mode | 27 | 1 |
| Application mode | 2 | 3 |
| Application mode | 10 | 10 |
| Application mode | 5 | 16 |
| Course | 33 | 12 |
| Father's occupation | 124 | 1 |
| Father's occupation | 121 | 1 |
| Father's occupation | 195 | 1 |
| Father's occupation | 161 | 1 |
| Father's occupation | 132 | 1 |
| Father's occupation | 131 | 1 |
| Father's occupation | 143 | 1 |
| Father's occupation | 134 | 1 |
| Father's occupation | 153 | 1 |
| Father's occupation | 174 | 1 |
| Father's occupation | 141 | 1 |
| Father's occupation | 114 | 1 |
| Father's occupation | 101 | 1 |
| Father's occupation | 154 | 1 |
| Father's occupation | 151 | 2 |
| Father's occupation | 182 | 2 |
| Father's occupation | 194 | 2 |
| Father's occupation | 172 | 2 |
| Father's occupation | 122 | 2 |
| Father's occupation | 102 | 2 |
| Father's occupation | 112 | 2 |
| Father's occupation | 135 | 3 |
| Father's occupation | 152 | 3 |
| Father's occupation | 183 | 3 |
| Father's occupation | 181 | 3 |
| Father's occupation | 123 | 3 |
| Father's occupation | 175 | 4 |
| Father's occupation | 103 | 4 |
| Father's occupation | 163 | 5 |
| Father's occupation | 192 | 6 |
| Father's occupation | 144 | 8 |
| Father's occupation | 171 | 8 |
| Father's occupation | 193 | 15 |
| Father's occupation | 99 | 19 |
| Father's qualification | 20 | 1 |
| Father's qualification | 13 | 1 |
| Father's qualification | 18 | 1 |
| Father's qualification | 42 | 1 |
| Father's qualification | 25 | 1 |
| Father's qualification | 44 | 1 |
| Father's qualification | 33 | 1 |
| Father's qualification | 27 | 1 |
| Father's qualification | 31 | 1 |
| Father's qualification | 26 | 2 |
| Father's qualification | 35 | 2 |
| Father's qualification | 41 | 2 |
| Father's qualification | 43 | 2 |
| Father's qualification | 10 | 2 |
| Father's qualification | 6 | 2 |
| Father's qualification | 29 | 3 |
| Father's qualification | 14 | 4 |
| Father's qualification | 22 | 4 |
| Father's qualification | 30 | 4 |
| Father's qualification | 9 | 5 |
| Father's qualification | 40 | 5 |
| Father's qualification | 36 | 8 |
| Father's qualification | 11 | 10 |
| Father's qualification | 5 | 18 |
| Marital Status | 3 | 4 |
| Marital Status | 6 | 6 |
| Mother's occupation | 125 | 1 |
| Mother's occupation | 173 | 1 |
| Mother's occupation | 131 | 1 |
| Mother's occupation | 171 | 1 |
| Mother's occupation | 152 | 2 |
| Mother's occupation | 122 | 2 |
| Mother's occupation | 153 | 2 |
| Mother's occupation | 151 | 3 |
| Mother's occupation | 132 | 3 |
| Mother's occupation | 143 | 3 |
| Mother's occupation | 134 | 4 |
| Mother's occupation | 10 | 4 |
| Mother's occupation | 193 | 4 |
| Mother's occupation | 175 | 5 |
| Mother's occupation | 192 | 5 |
| Mother's occupation | 144 | 6 |
| Mother's occupation | 123 | 7 |
| Mother's occupation | 141 | 8 |
| Mother's occupation | 194 | 11 |
| Mother's occupation | 99 | 17 |
| Mother's qualification | 18 | 1 |
| Mother's qualification | 22 | 1 |
| Mother's qualification | 27 | 1 |
| Mother's qualification | 26 | 1 |
| Mother's qualification | 44 | 1 |
| Mother's qualification | 14 | 2 |
| Mother's qualification | 11 | 3 |
| Mother's qualification | 30 | 3 |
| Mother's qualification | 35 | 3 |
| Mother's qualification | 36 | 3 |
| Mother's qualification | 10 | 3 |
| Mother's qualification | 29 | 3 |
| Mother's qualification | 42 | 4 |
| Mother's qualification | 6 | 4 |
| Mother's qualification | 43 | 4 |
| Mother's qualification | 41 | 6 |
| Mother's qualification | 9 | 8 |
| Mother's qualification | 39 | 8 |
| Mother's qualification | 40 | 9 |
| Nationality | 13 | 1 |
| Nationality | 32 | 1 |
| Nationality | 109 | 1 |
| Nationality | 108 | 1 |
| Nationality | 14 | 1 |
| Nationality | 17 | 1 |
| Nationality | 62 | 2 |
| Nationality | 25 | 2 |
| Nationality | 21 | 2 |
| Nationality | 101 | 2 |
| Nationality | 2 | 2 |
| Nationality | 105 | 2 |
| Nationality | 103 | 3 |
| Nationality | 11 | 3 |
| Nationality | 100 | 3 |
| Nationality | 24 | 5 |
| Nationality | 6 | 13 |
| Nationality | 22 | 13 |
| Nationality | 26 | 14 |
| Previous qualification | 5 | 1 |
| Previous qualification | 14 | 1 |
| Previous qualification | 15 | 2 |
| Previous qualification | 10 | 4 |
| Previous qualification | 43 | 6 |
| Previous qualification | 38 | 7 |
| Previous qualification | 4 | 8 |
| Previous qualification | 9 | 11 |
| Previous qualification | 6 | 16 |

## Lưu ý cho tiền xử lý
- Dùng danh sách nhóm biến trong src/config.py.
- Biến phân loại mã hóa số không phải biến liên tục.
- Cân nhắc các mã hiếm theo ngưỡng min_frequency=20.
- Kiểm tra các cặp biến tương quan cao.
- Không xóa ngoại lệ chỉ dựa trên quy tắc IQR.
- Không sửa data/raw/data.csv.
- Fit các bước tiền xử lý trên tập train.
