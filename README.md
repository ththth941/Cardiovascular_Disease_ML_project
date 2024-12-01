`READEME.md`
---------------------------
A readme file should be you guide to your project for others that may use it (for reproducibility sake) as well as for your future self after you've stepped away from the project for 6 months, and return having forgotten most of what you've done. At minimum, this markdown document should describe:

What the project does
Why the project is useful (for tools) or what you learned (for analysis projects)
How users can get started with the project
For research, a link to any papers or further documentation

`requirements.txt`
--------------------------
Trong văn bản này chứa các thư viện cần install và import vào các file huấn luyện mô hình, FastAPI và Streamlit để triển khai mô hình đã được huấn luyện.

`data/`
-------------------------
File data chứa dataset raw Cardiovascular_Disease_Dataset.csv 

`jupyer_notebook/`
-------------------------
File jupyter_notebook gồm 2 file, Cardiovascular_Disease_EDA.ipynb dùng để khám phá, nghiên cứu dữ liệu từ dataset và model-train.ipynb dùng để huấn luyện mô hình và lưu vào file `model/`

`model/`
------------------------
File model được dùng để lưu trữ mô hình máy học sau khi đã được huấn luyện từ file model-train.ipynb

`reports/`
------------------------
File chứa các báo cáo về kết quả đánh giá của các mô hình

`scripts/`
-----------------------
File thực hiện các bước như tiền xử lý dữ liệu, huấn luyện mô hình, tuning mô hình, đánh giá mô hình, lưu mô hình có kết quả đánh giá tốt nhất 
Run file main_scripts.py 

`code/`
----------------------
File api.py và app.py dùng để triển khai mô hình đã được huấn luyện thành 1 Web Application Dự đoán bệnh tim bằng Streamlit và FastAPI
Chạy file api.py trước -> gõ vào terminal "uvicorn code.api:app --reload"
Chạy file app.py --> gõ vào terminal "streamlit run code/app.py"

`src/`
---------------------
*code:
  *pycache: Thư mục chứa các file bytecode được biên dịch từ các file Python để tăng tốc độ thực thi.
  *api.cpython-312.pyc: File bytecode của module api.py.
  *api.py: File chứa các hàm và lớp để tương tác với API hoặc các dịch vụ bên ngoài.
  *app.py: File dùng để chạy ứng dụng Web Dự đoán bệnh tim bằng streamlit.
  *Chạy file api.py trước -> gõ vào terminal "uvicorn src.code.api:app --reload"
  *Chạy file app.py --> gõ vào terminal "streamlit run src/code/app.py"
data: Chứa các tập dữ liệu sử dụng để huấn luyện và đánh giá mô hình.
  Cardiovascular_Disease_Dataset.csv: File dữ liệu chính chứa thông tin về bệnh nhân.
jupyter_notebook: Chứa các notebook Jupyter để thực hiện EDA, xây dựng và đánh giá mô hình.
  Cardiovascular_Disease_EDA.ipynb: Notebook thực hiện phân tích dữ liệu khám phá (EDA).
  model-train.ipynb: Notebook huấn luyện mô hình.
model: Chứa các file mô hình đã được huấn luyện.
  mean_std_values_ML.pkl: File chứa các giá trị trung bình và độ lệch chuẩn để chuẩn hóa dữ liệu.
  model_ML.pkl: File chứa mô hình đã được huấn luyện.
model_training: Chứa các script Python liên quan đến quá trình huấn luyện mô hình.
  model_train.py: Script thực hiện quá trình huấn luyện mô hình.

