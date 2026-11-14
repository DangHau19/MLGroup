# 2. Related Work

**Phần E — Hoàng Đăng Hậu, MSSV 3124411085. Bản nháp.** Các trích dẫn dưới đây do E soạn theo hiểu biết về từng tài liệu; A cần mở lại từng nguồn để kiểm tra số trang/DOI và định dạng trước khi gộp vào mục 8 (xem danh sách cuối file).

## 2.1. Phân loại thể loại nhạc

Phân loại thể loại nhạc tự động là một bài toán kinh điển của truy vấn thông tin âm nhạc (MIR). Công trình thường được nhắc đến nhất là của Tzanetakis và Cook [1], dùng đặc trưng âm sắc, nhịp điệu và cao độ trích từ tín hiệu âm thanh rồi phân loại bằng các mô hình thống kê, đồng thời giới thiệu bộ dữ liệu GTZAN. Về sau, hướng làm chuyển dần sang mạng nơ-ron học biểu diễn trực tiếp từ phổ âm thanh [2]. Việc đánh giá cũng được xem xét lại: Sturm chỉ ra GTZAN có lỗi (trùng lặp, gán nhãn sai) làm sai lệch các kết quả được báo cáo [3]. Các bộ dữ liệu lớn như Million Song Dataset [4] cung cấp siêu dữ liệu và đặc trưng đã trích sẵn cho hàng triệu bản ghi, nên cho phép làm việc mà không cần tệp âm thanh.

Bài toán của nhóm thuộc dạng thứ hai: đầu vào là **bảng đặc trưng** mức bản ghi (danceability, energy, loudness, valence, tempo, ... cùng tên nghệ sĩ và tên bài) chứ không phải tín hiệu âm thanh. Các đặc trưng kiểu này được nhà cung cấp dịch vụ nhạc tính sẵn và mô tả trong tài liệu API của họ [5]; vì vậy mô hình học trên dữ liệu bảng, không cần nhánh xử lý âm thanh.

Một lưu ý phương pháp luận từ MIR rất liên quan đến bài này: khi cùng một nghệ sĩ xuất hiện ở cả tập huấn luyện và tập kiểm tra, mô hình có thể "nhận ra nghệ sĩ" thay vì học thể loại. Pampalk và cộng sự đề xuất *artist filter* (không để một nghệ sĩ nằm ở cả hai phía) [6], và Flexer phân tích kỹ ảnh hưởng của nó lên độ chính xác [7]. Trong bộ dữ liệu của nhóm, khoảng 63,8% dòng test có nghệ sĩ đã thấy ở train (mục 3), nên thông tin nghệ sĩ là hợp lệ và hữu ích cho bài thi này; tuy nhiên điều đó có nghĩa kết quả không phản ánh khả năng tổng quát hóa sang nghệ sĩ hoàn toàn mới.

## 2.2. Gradient boosting trên dữ liệu bảng

Với dữ liệu bảng, các mô hình cây tăng cường gradient [8] như XGBoost [9], LightGBM [10] và CatBoost [11] thường là lựa chọn mạnh. Grinsztajn và cộng sự báo cáo rằng trên nhiều bộ dữ liệu bảng cỡ vừa, mô hình dựa trên cây vẫn vượt các mạng nơ-ron học sâu [12]. CatBoost còn được thiết kế để xử lý đặc trưng phân loại và giảm thiên lệch của target statistics khi huấn luyện [11]. Đây là lý do nhóm dùng ba mô hình boosting làm trụ cột. Bên cạnh đó, các mô hình khác họ như ExtraTrees [13], SVM và MLP đóng vai trò tạo sự đa dạng cho ensemble.

Với biến có nhiều giá trị như tên nghệ sĩ, một cách phổ biến là *target encoding*: thay giá trị phân loại bằng thống kê của nhãn, có làm mịn để giảm nhiễu ở nhóm ít mẫu [14]. Vì thống kê này dùng nhãn nên phải tính trong từng fold (và OOF lồng bên trong) để tránh rò rỉ; phần này do C và D thực hiện (mục 4, 5).

## 2.3. Ensemble và stacking

Ý tưởng kết hợp nhiều mô hình để giảm sai số là nền tảng của học ensemble [15]. Có ba cách thường dùng. Thứ nhất là **trung bình (có trọng số) xác suất**, đơn giản nhưng chỉ có lợi khi các mô hình đủ tốt và đủ khác nhau. Thứ hai là **chọn mô hình có lặp** từ một thư viện mô hình (ensemble selection) [16]. Thứ ba là **stacking**: huấn luyện một meta-learner trên dự đoán ngoài mẫu (OOF) của các mô hình cơ sở [17], với biến thể dùng tổ hợp tuyến tính không âm của các mô hình [18]. Điều kiện then chốt của stacking là meta-learner chỉ được học trên dự đoán OOF; nếu dùng dự đoán trong mẫu, nó sẽ ưu tiên mô hình overfit.

Phần E (mục 6) triển khai cả ba cách trên bảy họ mô hình do D bàn giao và đánh giá bằng quy trình lồng (nested) để tránh tự đánh giá quá lạc quan.

## Tài liệu tham khảo của mục 2 (để A gộp vào mục 8)

[1] G. Tzanetakis and P. Cook, "Musical genre classification of audio signals," *IEEE Transactions on Speech and Audio Processing*, vol. 10, no. 5, pp. 293–302, 2002.

[2] K. Choi, G. Fazekas, K. Cho, and M. Sandler, "A tutorial on deep learning for music information retrieval," arXiv:1709.04396, 2017.

[3] B. L. Sturm, "The GTZAN dataset: Its contents, its faults, their effects on evaluation, and its future use," arXiv:1306.1461, 2013.

[4] T. Bertin-Mahieux, D. P. W. Ellis, B. Whitman, and P. Lamere, "The Million Song Dataset," in *Proc. 12th Int. Society for Music Information Retrieval Conf. (ISMIR)*, 2011.

[5] Spotify for Developers, "Web API reference: Get Track's Audio Features," https://developer.spotify.com/documentation/web-api/reference/get-audio-features (A kiểm tra link còn truy cập được và ghi ngày truy cập).

[6] E. Pampalk, A. Flexer, and G. Widmer, "Improvements of audio-based music similarity and genre classification," in *Proc. ISMIR*, 2005.

[7] A. Flexer, "A closer look on artist filters for musical genre classification," in *Proc. ISMIR*, 2007.

[8] J. H. Friedman, "Greedy function approximation: A gradient boosting machine," *Annals of Statistics*, vol. 29, no. 5, pp. 1189–1232, 2001.

[9] T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in *Proc. 22nd ACM SIGKDD*, 2016, pp. 785–794.

[10] G. Ke et al., "LightGBM: A highly efficient gradient boosting decision tree," in *Advances in Neural Information Processing Systems 30*, 2017.

[11] L. Prokhorenkova, G. Gusev, A. Vorobev, A. V. Dorogush, and A. Gulin, "CatBoost: Unbiased boosting with categorical features," in *Advances in Neural Information Processing Systems 31*, 2018.

[12] L. Grinsztajn, E. Oyallon, and G. Varoquaux, "Why do tree-based models still outperform deep learning on typical tabular data?" in *NeurIPS Datasets and Benchmarks Track*, 2022.

[13] P. Geurts, D. Ernst, and L. Wehenkel, "Extremely randomized trees," *Machine Learning*, vol. 63, no. 1, pp. 3–42, 2006.

[14] D. Micci-Barreca, "A preprocessing scheme for high-cardinality categorical attributes in classification and prediction problems," *ACM SIGKDD Explorations*, vol. 3, no. 1, pp. 27–32, 2001.

[15] T. G. Dietterich, "Ensemble methods in machine learning," in *Multiple Classifier Systems*, LNCS 1857, Springer, 2000, pp. 1–15.

[16] R. Caruana, A. Niculescu-Mizil, G. Crew, and A. Ksikes, "Ensemble selection from libraries of models," in *Proc. 21st ICML*, 2004.

[17] D. H. Wolpert, "Stacked generalization," *Neural Networks*, vol. 5, no. 2, pp. 241–259, 1992.

[18] L. Breiman, "Stacked regressions," *Machine Learning*, vol. 24, no. 1, pp. 49–64, 1996.
