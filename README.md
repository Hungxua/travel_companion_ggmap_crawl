Thuật toán được sử dụng trong 
route\main.py
 là Greedy Algorithm (Thuật toán tham lam) kết hợp với hệ thống Weighted Scoring (Chấm điểm có trọng số).

Dưới đây là giải thích chi tiết về cách thuật toán hoạt động:

1. Tổng quan phương pháp
Thuật toán xây dựng lịch trình theo từng bước (step-by-step). Tại mỗi bước (mỗi khi chọn địa điểm tiếp theo), nó sẽ tính toán và chọn địa điểm "tốt nhất" có thể đi từ vị trí hiện tại, dựa trên nhiều tiêu chí như sở thích, khoảng cách, và thời gian.

2. Hệ thống chấm điểm (Scoring System)
Mỗi địa điểm (Place) được chấm điểm dựa trên sự phù hợp với người dùng. Hàm 
calculate_place_score
 tính điểm dựa trên các yếu tố sau:

Sở thích (Interests) (40%): Nếu địa điểm có "vibe" trùng với sở thích người dùng (ví dụ: "healing", "photography"), điểm sẽ cao hơn.
Bạn đồng hành (Companion) (20%): Điểm thưởng nếu địa điểm phù hợp với loại nhóm đi cùng (ví dụ: couple, family).
Độ phổ biến (Popularity) (20%): Dựa trên Rating và số lượng Review trên Google Maps.
Ưu tiên (Priority) (10%): Điểm ưu tiên có sẵn trong dữ liệu.
Must-visit (10%): Điểm thưởng lớn nếu địa điểm được đánh dấu là "phải đi".
3. Thuật toán chọn địa điểm (Greedy Strategy)
Hàm 
greedy_select_attractions
 thực hiện việc chọn điểm tham quan:

Lọc ứng viên: Từ vị trí hiện tại, xem xét tất cả các điểm chưa đi.
Tính điểm ứng viên: Hàm 
calculate_candidate_score
 tính điểm cho mỗi ứng viên dựa trên:
Quality Score: Điểm chất lượng (từ bước 2 ở trên).
Distance Penalty: Phạt điểm nếu khoảng cách quá xa (để tối ưu di chuyển).
Time Penalty: Phạt điểm nếu tốn quá nhiều thời gian.
Chọn tốt nhất (Greedy choice): Sắp xếp các ứng viên theo điểm tổng hợp (combined_score) và chọn địa điểm có điểm cao nhất.
Cập nhật:
Đánh dấu địa điểm đã đi.
Cập nhật vị trí hiện tại là địa điểm vừa chọn.
Trừ quỹ thời gian trong ngày.
Lặp lại: Tiếp tục chọn cho đến khi hết thời gian hoặc đủ số lượng địa điểm (tối đa 4 điểm/ngày).
4. Xử lý logic phụ trợ
Ngoài thuật toán chính, code còn có các logic rule-based (dựa trên luật):

Chỗ ở (Accommodation):
Quy tắc 20km: Nếu điểm cuối ngày cách điểm đầu ngày quá 20km -> Tìm khách sạn mới ở gần điểm cuối. Nếu không -> Quay về khách sạn cũ hoặc tìm quanh điểm đầu.
Ăn uống (Meals):
Dùng thuật toán Nearest Neighbor (Láng giềng gần nhất): Tìm quán ăn phù hợp (sáng/trưa/tối) gần nhất với vị trí hiện tại vào giờ ăn.
Ràng buộc thời gian:
Giới hạn mỗi ngày đi tối đa 480 phút (8 tiếng).
Tính toán thời gian di chuyển giữa các điểm (có hệ số phạt cho đường núi).
Tóm lại
Thuật toán này không đảm bảo tìm ra lịch trình hoàn hảo nhất toàn cục (Global Optimum), nhưng nó đảm bảo tìm ra lịch trình tốt và hợp lý (Local Optimum) một cách nhanh chóng, cân bằng được giữa sở thích người dùng và sự thuận tiện trong di chuyển.

