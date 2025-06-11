from setuptools import setup, find_packages

setup(
    name="NDC_framework",                           # Đặt tên cho package của bạn
    version="0.1.0",                               # Phiên bản
    description="researching",
    author="Than HUYNH VAN",
    author_email="hthan401@gmail.com",
    package_dir={"": "src"},                       # Đây là điểm quan trọng! 
    packages=find_packages(where="src"),           # Tự động tìm packages trong src/
    install_requires=[
        # (Liệt kê các thư viện phụ thuộc ở đây, nếu có, ví dụ:)
        # 'numpy',
        # 'torch',
    ],
    python_requires='>=3.10',                 # Chọn phiên bản Python tối thiểu
    include_package_data=True,
)