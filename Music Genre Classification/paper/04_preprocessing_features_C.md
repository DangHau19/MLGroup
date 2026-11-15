# 4. Preprocessing & Feature Extraction

**Phần C — Phan Lê Minh Huy, MSSV: 3124411111.** 
Nội dung trình bày quy trình tiền xử lý dữ liệu và xây dựng đặc trưng phục vụ bài toán phân loại thể loại nhạc (Music Genre Classification). Quy trình chính thức được triển khai trong `code/features.py`, `code/artist_target_encoder_C.py` và `code/02_ablation_foldsafe_C.py`; `code/02_ablation_target_C.py` là lệnh tương thích, chuyển tiếp tới runner fold-safe. Notebook `code/02_features_C.ipynb` kiểm tra dữ liệu đầu ra, chạy ablation fold-safe và trực quan hóa các phiên bản đặc trưng.

Toàn bộ quá trình sử dụng dữ liệu do cuộc thi Kaggle cung cấp, không bổ sung dữ liệu nghệ sĩ, tên bài hát hoặc nhãn thể loại từ các nguồn bên ngoài. Kết quả được đánh giá bằng Stratified 5-Fold Cross-Validation với seed 42, sử dụng Accuracy làm tiêu chí chính và Macro-F1 làm tiêu chí bổ sung.

## 4.1. Mục tiêu và cấu trúc quy trình tiền xử lý

Bài toán Music Genre Classification yêu cầu dự đoán nhãn `Class` thuộc 11 lớp từ 0 đến 10. Dữ liệu huấn luyện gồm **14.396 dòng × 18 cột**, tập kiểm tra gồm **3.600 dòng × 17 cột**. Sau khi loại bỏ `Id` và nhãn `Class`, dữ liệu có 16 thuộc tính đầu vào, bao gồm hai biến dạng chuỗi là `Artist Name`, `Track Name` cùng 14 biến số hoặc mã rời rạc liên quan đến đặc điểm âm nhạc.

Các biến số bao gồm `Popularity`, `danceability`, `energy`, `key`, `loudness`, `mode`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `duration_in min/ms` và `time_signature`. Đây là dữ liệu dạng bảng, không có tín hiệu âm thanh trực tiếp, vì vậy phương pháp xử lý tập trung vào đặc trưng số, đặc trưng tương tác và thông tin nghệ sĩ thay vì trích xuất phổ âm thanh hoặc sử dụng mạng CNN.

Phân tích EDA cho thấy dữ liệu tồn tại một số vấn đề đáng chú ý: mất cân bằng nhãn, giá trị thiếu có quy luật, thời lượng bài hát sử dụng hai thang đo khác nhau và nhiều nghệ sĩ xuất hiện lặp lại giữa các bản ghi. Những đặc điểm này ảnh hưởng trực tiếp đến việc lựa chọn phương pháp tiền xử lý và xây dựng đặc trưng.

Để đánh giá tác động của từng kỹ thuật, phần C phát triển ba phiên bản đặc trưng v0, v1, v2 và một cấu hình mở rộng v2 kết hợp Artist Target Encoding. Các phiên bản được triển khai thông qua hàm `build_features(train, test, version)`, giữ nguyên giao diện đầu ra gồm `X`, `y`, `X_test`, `test_ids` và `classes` để tương thích với quy trình huấn luyện chung của nhóm.

Quy trình xử lý gồm các công đoạn chính: chuẩn bị dữ liệu, xử lý giá trị thiếu, chuẩn hóa thời lượng, xây dựng đặc trưng tần suất, biến đổi phi tuyến, kết hợp đặc trưng, mã hóa thông tin nghệ sĩ và đánh giá bằng cross-validation. Cột `Id` chỉ được sử dụng để bảo toàn thứ tự dữ liệu và xuất kết quả dự đoán, không được đưa vào mô hình như một biến giải thích.

## 4.2. Xử lý dữ liệu thiếu và chuẩn hóa thời lượng

### 4.2.1. Xử lý dữ liệu thiếu

Trong tập huấn luyện, ba đặc trưng có giá trị thiếu đáng chú ý là `instrumentalness` (24,60%), `key` (11,18%) và `Popularity` (2,31%). Tổng số ô thiếu được ghi nhận trong quá trình kiểm tra dữ liệu là **5.483**.

Việc thay thế trực tiếp toàn bộ giá trị thiếu có thể làm mất thông tin về tình trạng thiếu dữ liệu. Đặc biệt, phân tích EDA cho thấy tỷ lệ thiếu `instrumentalness` không giống nhau giữa các lớp. Do đó, bên cạnh phương pháp điền giá trị thiếu bằng trung vị, phiên bản v1 và v2 bổ sung các cờ nhị phân để ghi nhận trạng thái thiếu của từng quan sát.

Các biến mới gồm `Popularity_missing`, `key_missing` và `instrumentalness_missing`. Mỗi biến nhận giá trị 1 khi thuộc tính tương ứng bị thiếu và 0 trong trường hợp có dữ liệu.

Đối với phép điền trung vị, giá trị thống kê được tính từ phần dữ liệu huấn luyện tương ứng. Trong lần thực nghiệm CV cuối, các phiên bản đặc trưng được xây dựng lại theo từng outer training fold thay vì tạo một ma trận đặc trưng duy nhất trước khi chia dữ liệu. Cách thực hiện này giúp hạn chế việc sử dụng thống kê từ outer validation để xử lý dữ liệu huấn luyện.

Riêng `key`, dữ liệu quan sát có giá trị từ 1 đến 11 và không xuất hiện mã 0. Tuy nhiên, chưa có cơ sở xác nhận toàn bộ NaN của `key` tương ứng với nốt C. Vì vậy, phần C không tự động gán các giá trị thiếu thành 0 mà sử dụng cờ thiếu kết hợp với phương pháp điền giá trị đã xác định.

### 4.2.2. Chuẩn hóa đơn vị thời lượng

Đặc trưng `duration_in min/ms` có phân phối thể hiện hai nhóm giá trị khác biệt. Theo EDA, có 2.075 dòng train (14,41%) có thời lượng dưới 30, trong khi phần lớn các dòng còn lại có giá trị hàng chục nghìn đến hàng trăm nghìn.

Dựa trên giả thuyết về sự không thống nhất đơn vị, nhóm xem các giá trị nhỏ là phút và các giá trị lớn là mili giây. File `features.py` sử dụng ngưỡng 1.000 để xác định nhóm thời lượng nhỏ. Trong dữ liệu được kiểm tra, ngưỡng dưới 30 và dưới 1.000 tạo ra cùng cách phân nhóm đối với cả tập train và test.

Phép quy đổi được triển khai như sau:

- Nếu `duration_in min/ms` nhỏ hơn 1.000, thời lượng được nhân với 60.000 để chuyển từ phút sang mili giây.
- Nếu giá trị từ 1.000 trở lên, dữ liệu được giữ nguyên theo giả định đơn vị mili giây.
- Biến `duration_is_min` được bổ sung để ghi nhận nhóm đơn vị ban đầu.
- Biến `duration_ms` lưu thời lượng sau quy đổi.

Cờ `duration_is_min` được giữ lại vì EDA cho thấy mối liên hệ đáng kể giữa nhóm đơn vị và nhãn `Class`, đặc biệt đối với các lớp 0, 3, 4 và 7. Tuy nhiên, đây có thể là tín hiệu liên quan đến nguồn thu thập dữ liệu thay vì phản ánh bản chất âm nhạc. Do chưa có metadata xác nhận đơn vị từng dòng, phép quy đổi được xem là giả định kỹ thuật cần nêu rõ khi diễn giải kết quả.

## 4.3. Xây dựng các phiên bản đặc trưng

### 4.3.1. Phiên bản v0 — Baseline Features

Phiên bản v0 sử dụng các đặc trưng số gốc sau khi loại bỏ cột định danh và nhãn. Giá trị thiếu được điền bằng trung vị của dữ liệu huấn luyện. Phiên bản này chưa xử lý hiện tượng lẫn đơn vị thời lượng, chưa bổ sung cờ thiếu dữ liệu và chưa khai thác thông tin từ tên nghệ sĩ hoặc tên bài hát.

Mục tiêu của v0 là tạo mốc đối chứng để đo lường hiệu quả của các phép biến đổi bổ sung trong v1, v2 và cấu hình Target Encoding.

### 4.3.2. Phiên bản v1 — Duration, Missing Indicators và Frequency Features

Phiên bản v1 kế thừa các biến số ban đầu, đồng thời bổ sung các đặc trưng phản ánh tình trạng thiếu dữ liệu, thang đo thời lượng và tần suất xuất hiện của nghệ sĩ, bài hát.

Đối với thông tin nghệ sĩ, `artist_count` biểu diễn số lần xuất hiện của một nghệ sĩ trong dữ liệu được xử lý. Tương tự, `track_count` biểu diễn số lần xuất hiện của tên bài hát. Hai đặc trưng này không sử dụng nhãn `Class` nên thuộc nhóm thống kê không giám sát.

Ngoài ra, `artist_in_train` là cờ cho biết nghệ sĩ có xuất hiện trong tập huấn luyện được truyền vào hàm xây dựng đặc trưng hay không. Vì ý nghĩa của biến này có thể khác nhau giữa dữ liệu huấn luyện và validation/test, kết quả cần được diễn giải thận trọng.

Việc bổ sung các đặc trưng trên giúp mô hình khai thác thông tin mà phiên bản v0 chưa biểu diễn trực tiếp, đồng thời giữ lại dấu hiệu liên quan đến trạng thái thiếu dữ liệu và nguồn đơn vị thời lượng.

### 4.3.3. Phiên bản v2 — Nonlinear Transformations và Interaction Features

Phiên bản v2 kế thừa v1 và bổ sung các phép biến đổi nhằm biểu diễn rõ hơn những quan hệ phi tuyến giữa các đặc trưng âm nhạc.

**Biến đổi logarit:** Các biến `speechiness`, `instrumentalness`, `liveness` và `acousticness` được biến đổi bằng `log1p` sau khi giới hạn dưới tại 0. Phương pháp này hỗ trợ giảm độ lệch phân phối của các biến không âm và xử lý được giá trị bằng 0.

**Giới hạn loudness:** Đặc trưng `loudness_clipped` được tạo bằng cách giới hạn `loudness` trong khoảng [-40, 5]. Đây là đặc trưng bổ sung, không thay thế hoàn toàn biến loudness gốc. Theo các cực trị được ghi nhận trong EDA, tác động của phép cắt đuôi có thể không lớn trên dữ liệu hiện tại.

**Đặc trưng tỷ số:** Biến `energy_acousticness_ratio` được tính bằng tỷ số giữa `energy` và `acousticness` sau khi bổ sung hằng số ổn định 0,01 vào mẫu số. Phép biến đổi nhằm mô tả quan hệ giữa mức năng lượng và tính acoustic của bài hát, đồng thời tránh trường hợp chia cho 0.

**Đặc trưng tương tác:** Hai biến `danceability_x_valence` và `loudness_x_energy` được xây dựng bằng phép nhân giữa các thuộc tính tương ứng. Những đặc trưng này cho phép mô hình khai thác mối quan hệ kết hợp giữa khả năng nhảy, sắc thái cảm xúc, âm lượng và năng lượng.

**Kết hợp key và mode:** Đặc trưng `key_mode_24` được tính từ `key + 12 × mode` để biểu diễn tổ hợp giữa cao độ và chế độ trưởng/thứ. Khi `key` bị thiếu, đặc trưng kết hợp nhận giá trị -1 nhằm giữ lại trạng thái không xác định.

**Nhóm tempo:** Hai đặc trưng `tempo_rounded_10` và `tempo_group_20` được bổ sung. Đặc trưng thứ nhất làm tròn tempo theo bước 10 BPM; đặc trưng thứ hai chia tempo thành các nhóm có độ rộng 20 BPM. Cách biểu diễn này giúp mô hình tiếp cận tốc độ âm nhạc ở cả dạng liên tục lẫn dạng nhóm.

### Bảng 1. Danh sách các nhóm đặc trưng được xây dựng

| Nhóm đặc trưng | Biến đại diện | Phương pháp xử lý |
|---|---|---|
| Biến số ban đầu | danceability, energy, valence, tempo và các biến số khác | Giữ giá trị gốc |
| Thời lượng | duration_ms, duration_is_min | Quy đổi theo giả định đơn vị và giữ cờ |
| Dữ liệu thiếu | Popularity_missing, key_missing, instrumentalness_missing | Mã hóa nhị phân |
| Tần suất | artist_count, track_count | Đếm số lần xuất hiện |
| Nghệ sĩ đã biết | artist_in_train | Kiểm tra xuất hiện trong train |
| Log transformation | speechiness_log1p, instrumentalness_log1p, liveness_log1p, acousticness_log1p | log1p |
| Loudness | loudness_clipped | Giới hạn [-40, 5] |
| Tỷ số | energy_acousticness_ratio | Chia với hằng số ổn định |
| Tương tác | danceability_x_valence, loudness_x_energy | Phép nhân |
| Tổ hợp giọng | key_mode_24 | Kết hợp key và mode |
| Nhóm tốc độ | tempo_rounded_10, tempo_group_20 | Làm tròn và chia nhóm |
| Target Encoding | artist_te_0 đến artist_te_10 | Xác suất lớp có làm mịn |

Khi chạy phiên bản v2 trên dữ liệu Kaggle, ma trận đặc trưng thu được có kích thước **14.396 × 32** đối với train và **3.600 × 32** đối với test. Hai ma trận có cùng thứ tự cột và không chứa giá trị NaN sau khi tiền xử lý.

## 4.4. Artist Target Encoding và kiểm soát rò rỉ dữ liệu

### 4.4.1. Cơ sở lựa chọn Artist Target Encoding

Theo EDA, có 2.296/3.600 dòng test (63,78%) chứa nghệ sĩ đã xuất hiện trong train. Ngoài ra, trong kiểm chứng 5-fold, 58,68% mẫu validation có nghệ sĩ xuất hiện trong train fold. Quy tắc dự đoán theo lớp phổ biến nhất của nghệ sĩ đạt accuracy 62,31% trên riêng nhóm nghệ sĩ đã biết.

Những kết quả này cho thấy thông tin nghệ sĩ có khả năng hỗ trợ đáng kể cho bài toán phân loại thể loại nhạc. Tuy nhiên, việc trực tiếp mã hóa nghệ sĩ theo nhãn của toàn bộ tập train trước khi chia cross-validation sẽ gây rò rỉ nhãn, đặc biệt khi một nghệ sĩ xuất hiện trong nhiều bản ghi.

Vì vậy, phần C xây dựng `ArtistTargetEncoder` theo giao diện `fit` và `transform` của scikit-learn, kết hợp kỹ thuật làm mịn và tạo đặc trưng out-of-fold.

### 4.4.2. Phương pháp mã hóa theo xác suất lớp

Đối với mỗi nghệ sĩ, transformer ước lượng phân phối xác suất của 11 lớp nhạc dựa trên các nhãn thuộc tập dữ liệu được sử dụng để fit.

Phương pháp làm mịn sử dụng công thức:

\[
TE(a,k)=\frac{n_{a,k}+m p_k}{n_a+m}
\]

Trong đó:

- \(TE(a,k)\) là giá trị mã hóa của nghệ sĩ \(a\) đối với lớp \(k\).
- \(n_{a,k}\) là số quan sát của nghệ sĩ \(a\) thuộc lớp \(k\).
- \(n_a\) là tổng số quan sát của nghệ sĩ \(a\).
- \(p_k\) là tỷ lệ nền của lớp \(k\) trong dữ liệu được dùng để fit.
- \(m=20\) là hệ số làm mịn được sử dụng trong thực nghiệm.

Khi nghệ sĩ có ít mẫu, kết quả mã hóa được điều chỉnh về gần phân phối prior, giúp hạn chế tác động của những tần suất lớp không ổn định. Với nghệ sĩ chưa từng xuất hiện trong tập fit, transformer sử dụng phân phối prior thay cho xác suất theo nghệ sĩ.

Mỗi nghệ sĩ được biểu diễn bằng 11 biến `artist_te_0` đến `artist_te_10`. Các giá trị này mô tả xác suất tương ứng với từng lớp và có tổng bằng 1 trong các phép kiểm tra đã thực hiện.

### 4.4.3. Nested Out-of-Fold Encoding

Để tránh việc nhãn của một bản ghi tham gia trực tiếp vào quá trình tạo đặc trưng Target Encoding của chính bản ghi đó, phần C triển khai nested out-of-fold encoding trong quy trình cross-validation.

Tập train được chia thành 5 outer folds theo `common.get_folds(y)`. Trong mỗi outer fold, phần huấn luyện tiếp tục được chia thành 4 inner folds bằng StratifiedKFold.

Với từng inner fold, encoder chỉ học thống kê nhãn từ inner training fold, sau đó biến đổi inner validation fold. Các kết quả được ghép lại thành ma trận Target Encoding OOF cho toàn bộ phần outer training.

Đối với outer validation và test, một encoder khác được fit trên toàn bộ outer training fold rồi thực hiện phép biến đổi tương ứng. Nhờ đó, mô hình không sử dụng trực tiếp nhãn của outer validation trong quá trình xây dựng đặc trưng Target Encoding.

Quy trình chính thức được triển khai trong `code/artist_target_encoder_C.py` và `code/02_ablation_foldsafe_C.py`. `ArtistTargetEncoder.fit_transform` tạo inner OOF cho dữ liệu train, còn `transform` dùng mapping đã fit trên outer train. `common.cross_validate_model` giữ nguyên DataFrame và chọn hàng bằng `.iloc`, nên encoder dùng được trong sklearn `Pipeline` qua CV chung. Ablation đầy đủ vẫn dùng runner fold-safe để fit các bước xây đặc trưng/điền median trong từng outer fold. Lệnh `code/02_ablation_target_C.py` chuyển tiếp tới runner này để không còn chạy luồng tiền xử lý cũ.

## 4.5. Thiết lập thực nghiệm và Ablation Study

Để đánh giá ảnh hưởng của từng nhóm đặc trưng, phần C sử dụng cố định mô hình `HistGradientBoostingClassifier` với `max_iter = 150`, `learning_rate = 0.08` và `random_state = 42`.

Các cấu hình được đánh giá bằng Stratified 5-Fold Cross-Validation, sử dụng cùng cách chia dữ liệu của nhóm. Accuracy được chọn làm metric chính theo quy ước của Challenge 2; Macro-F1 được tính bổ sung để phản ánh khả năng dự đoán đối với các lớp có quy mô khác nhau.

Thử nghiệm bao gồm bốn cấu hình: v0, v1, v2 và v2 + Artist Target Encoding. Phiên bản thực nghiệm cuối sử dụng hàm xây dựng đặc trưng theo từng outer training fold thay vì tiền xử lý một lần trên toàn bộ train trước khi chia CV.

Mã nguồn đánh giá được đặt tại `code/02_ablation_foldsafe_C.py`, bảng số liệu được lưu tại `evidence/cv_logs/ablation_foldsafe_C.csv`. Các kết quả dưới đây được lấy từ lần chạy thực tế trên bộ dữ liệu Kaggle của nhóm.

### Bảng 2. Kết quả Ablation Study với Stratified 5-Fold CV

| Cấu hình | CV Accuracy | Độ lệch chuẩn | Macro-F1 |
|---|---:|---:|---:|
| v0 | 0.536954 | 0.005811 | 0.572361 |
| v1 | 0.559114 | 0.006972 | 0.608375 |
| v2 | 0.559044 | 0.003184 | 0.608798 |
| **v2 + Artist Target Encoding** | **0.641081** | **0.005267** | **0.692182** |

*Độ lệch chuẩn được tính từ accuracy trên năm folds. Số liệu phản ánh lần chạy thử nghiệm đã hoàn thành trên môi trường Python của thành viên phụ trách phần C.*

Kết quả cho thấy v1 cải thiện accuracy từ 0,5370 lên 0,5591 so với v0, tương ứng mức tăng khoảng 2,22 điểm phần trăm. Điều này cho thấy các phép chuẩn hóa thời lượng, bổ sung cờ thiếu và xây dựng đặc trưng tần suất có ích trong lần chạy này.

Phiên bản v2 đạt accuracy 0,5590, gần như bằng v1 (thấp hơn khoảng 0,01 điểm phần trăm). Macro-F1 tăng nhẹ từ 0,6084 lên 0,6088. Như vậy, các phép biến đổi phi tuyến và tương tác được bổ sung chưa tạo cải thiện accuracy đáng kể khi sử dụng riêng lẻ.

Một nguyên nhân có thể là HistGradientBoosting đã có khả năng học các quan hệ phi tuyến từ đặc trưng gốc, khiến lợi ích của những biến tương tác thủ công bị hạn chế. Đây là cách diễn giải từ kết quả thực nghiệm, chưa phải kết luận nhân quả được chứng minh bằng phân tích importance riêng.

Khi kết hợp v2 với Artist Target Encoding, accuracy đạt **0,641081 (64,11%)**, Macro-F1 đạt **0,692182**. So với v1, cấu hình kết hợp cải thiện khoảng **8,20 điểm phần trăm accuracy**. So với v2, mức cải thiện cũng khoảng 8,20 điểm phần trăm.

Kết quả cho thấy việc mã hóa phân phối thể loại theo nghệ sĩ là hướng khai thác đặc trưng hiệu quả trong dữ liệu hiện tại. Tuy nhiên, bảng ablation chưa có cấu hình v1 + Target Encoding, do đó chưa thể tách riêng hoàn toàn tác động của Target Encoding và những phép biến đổi bổ sung trong v2 khi kết hợp.

### Hình 3. So sánh hiệu quả các phiên bản đặc trưng

Biểu đồ `paper/figures/ablation_C.png` được tạo trong notebook `02_features_C.ipynb` để so sánh CV Accuracy giữa các cấu hình. Cần đồng bộ biểu đồ với kết quả từ `ablation_foldsafe_C.csv`, vì hình được tạo trong thử nghiệm trước có thể sử dụng những điểm số sơ bộ khác với bảng kết quả cuối.

Dữ liệu cuối cho thấy thứ tự hiệu quả theo accuracy là: v2 + Artist Target Encoding cao nhất, tiếp theo là v1, v2 và v0.

## 4.6. Thảo luận và giới hạn phương pháp

Kết quả thực nghiệm cho thấy việc xử lý những đặc điểm riêng của bộ dữ liệu mang lại hiệu quả đáng kể. Cờ dữ liệu thiếu giúp lưu giữ trạng thái thiếu, trong khi phép quy đổi thời lượng và đặc trưng tần suất cung cấp thêm tín hiệu dự đoán mà biến số gốc chưa biểu diễn rõ.

Tuy nhiên, có một số giới hạn cần được xem xét khi diễn giải kết quả.

Thứ nhất, biến `duration_is_min` có thể phản ánh nguồn thu thập dữ liệu thay vì đặc tính âm nhạc. Nếu dữ liệu được triển khai trên một nguồn mới với đơn vị thời lượng thống nhất, hiệu quả của đặc trưng này có thể thay đổi.

Thứ hai, Artist Target Encoding khai thác mối quan hệ giữa nghệ sĩ và nhãn thể loại. Phương pháp này có thể đạt hiệu quả tốt hơn khi nghệ sĩ trong validation đã xuất hiện ở train, nhưng khả năng dự đoán nghệ sĩ mới chưa được đánh giá riêng bằng GroupKFold hoặc phương pháp chia nhóm theo nghệ sĩ.

Thứ ba, quy trình cuối đã thực hiện việc tạo đặc trưng theo outer training fold và nested OOF cho các thống kê sử dụng nhãn. Tuy nhiên, triển khai v1/v2 vẫn sử dụng một số thống kê không nhãn trên dữ liệu kết hợp giữa train fold, validation và test theo thiết kế ban đầu. Đây là cách tiếp cận có sử dụng thông tin phân phối của các mẫu cần dự đoán, vì vậy kết quả chưa tương đương với quy trình huấn luyện hoàn toàn độc lập trên train fold theo nghĩa nghiêm ngặt.

Thứ tư, các đặc trưng mới của v2 chưa được kiểm chứng riêng lẻ bằng phép loại bỏ từng biến hoặc phân tích permutation importance. Do đó, chưa có cơ sở khẳng định biến tương tác cụ thể nào mang lại lợi ích hoặc làm giảm hiệu quả mô hình.

Thứ năm, kết quả phản ánh hiệu quả của một cấu hình HistGradientBoosting với các siêu tham số cố định, không phải khẳng định mô hình hoặc tập đặc trưng tối ưu trên mọi thuật toán. Phần lựa chọn mô hình và điều chỉnh siêu tham số được thực hiện ở giai đoạn tiếp theo của nhóm.

Ngoài ra, việc sử dụng Accuracy làm tiêu chí chính có thể khiến kết quả chịu ảnh hưởng lớn của các lớp đông mẫu. Vì vậy, Macro-F1 cần được báo cùng Accuracy và đối chiếu thêm với confusion matrix để đánh giá mức độ sai lệch giữa các lớp.

## 4.7. Kết luận và bàn giao kết quả

Phần C đã xây dựng và kiểm chứng ba phiên bản đặc trưng v0, v1, v2, đồng thời phát triển Artist Target Encoding có làm mịn và nested out-of-fold encoding. Quy trình đánh giá được thực hiện bằng Stratified 5-Fold CV trên tập dữ liệu của cuộc thi, sử dụng mô hình HistGradientBoosting với cấu hình cố định.

Trong lần chạy fold-safe mới nhất, cấu hình **v2 + Artist Target Encoding đạt CV Accuracy 0,641081 và Macro-F1 0,692182**. Kết quả này cho thấy thông tin về nghệ sĩ có giá trị dự báo đáng kể đối với bài toán phân loại thể loại nhạc trong bộ dữ liệu hiện tại.

Các sản phẩm bàn giao bao gồm `code/features.py` với phiên bản v2, `code/artist_target_encoder_C.py`, script chính thức `code/02_ablation_foldsafe_C.py`, notebook `code/02_features_C.ipynb`, bảng kết quả `evidence/cv_logs/ablation_foldsafe_C.csv` và các tệp xác suất OOF/test phục vụ đánh giá mô hình. `code/02_ablation_target_C.py` được giữ làm tên lệnh tương thích và chuyển tiếp tới runner fold-safe.

Kết quả và đặc trưng của phần C là cơ sở để các thành viên tiếp tục thử nghiệm LightGBM, XGBoost, CatBoost và các phương pháp ensemble. Các giới hạn về nguồn đơn vị thời lượng, nghệ sĩ chưa biết và việc sử dụng thống kê không nhãn trên dữ liệu kết hợp cần được ghi nhận khi tổng hợp kết luận chung của nhóm.