# 🖼️ 영상처리 교육 자료 (Image Processing Education)

> C/C++, Python, Pillow, scikit-learn, OpenCV를 활용한 체계적인 영상처리 학습 가이드

---

## 📚 목차 (Table of Contents)

1. [C/C++을 이용한 영상처리 (RAW 흑백 이미지)](#1-cc를-이용한-영상처리-raw-흑백-이미지)
   - [화소점 처리](#11-화소점-처리-pixel-point-processing)
   - [기하학 처리](#12-기하학-처리-geometric-processing)
   - [화소영역 처리](#13-화소영역-처리-area-processing)
2. [Python을 이용한 영상처리 (RAW + RGB)](#2-python을-이용한-영상처리-raw--rgb)
   - [화소점 처리](#21-화소점-처리)
   - [기하학 처리](#22-기하학-처리)
   - [화소영역 처리](#23-화소영역-처리)
3. [Python + Pillow + 마우스 처리](#3-python--pillow--마우스-처리)
4. [Python + scikit-learn 머신러닝](#4-python--scikit-learn-머신러닝)
5. [Python + OpenCV 영상처리 및 검출](#5-python--opencv-영상처리-및-검출)

---

## 📁 프로젝트 구조

```
image-processing-education/
├── 01_cpp_raw/
│   ├── images/              # 테스트용 RAW 이미지
│   ├── pixel_point/         # 화소점 처리
│   ├── geometric/           # 기하학 처리
│   └── area/                # 화소영역 처리
├── 02_python_raw_rgb/
│   ├── pixel_point/
│   ├── geometric/
│   └── area/
├── 03_pillow_mouse/
│   └── roi_processing.py
├── 04_sklearn_ml/
│   ├── iris_classifier.py
│   ├── wine_classifier.py
│   └── fraud_detection.py
└── 05_opencv/
    ├── image_processing/
    └── detection/
```

---

## 🔧 개발 환경 설정

### C/C++ 환경
```bash
# Linux/Ubuntu
sudo apt-get install build-essential

# 컴파일 방법
gcc -o output program.c
g++ -o output program.cpp
```

### Python 환경
```bash
pip install numpy matplotlib pillow scikit-learn opencv-python
```

---

# 1. C/C++를 이용한 영상처리 (RAW 흑백 이미지)

> RAW 이미지는 헤더 없이 픽셀 값(0~255)만 저장된 바이너리 파일입니다.  
> 기본 이미지 크기: **256×256 (가로×세로)**

## RAW 이미지 기본 구조

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WIDTH  256
#define HEIGHT 256

// RAW 이미지 읽기
unsigned char* readRaw(const char* filename, int width, int height) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) { perror("파일 열기 실패"); return NULL; }
    
    unsigned char* img = (unsigned char*)malloc(width * height);
    fread(img, 1, width * height, fp);
    fclose(fp);
    return img;
}

// RAW 이미지 저장
void writeRaw(const char* filename, unsigned char* img, int width, int height) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) { perror("파일 저장 실패"); return; }
    fwrite(img, 1, width * height, fp);
    fclose(fp);
}
```

---

## 1.1 화소점 처리 (Pixel Point Processing)

화소점 처리는 **각 픽셀을 독립적으로** 변환하는 기법입니다.

### 📌 밝기 조절 (Brightness)

```c
// 밝게 만들기 (덧셈)
void brighten(unsigned char* src, unsigned char* dst, int size, int value) {
    for (int i = 0; i < size; i++) {
        int pixel = src[i] + value;
        dst[i] = (pixel > 255) ? 255 : (unsigned char)pixel;  // 클리핑
    }
}

// 어둡게 만들기 (뺄셈)
void darken(unsigned char* src, unsigned char* dst, int size, int value) {
    for (int i = 0; i < size; i++) {
        int pixel = src[i] - value;
        dst[i] = (pixel < 0) ? 0 : (unsigned char)pixel;
    }
}

// 사용 예시
int main() {
    unsigned char* src = readRaw("input.raw", WIDTH, HEIGHT);
    unsigned char* dst = (unsigned char*)malloc(WIDTH * HEIGHT);
    
    brighten(src, dst, WIDTH * HEIGHT, 50);   // 50만큼 밝게
    writeRaw("bright.raw", dst, WIDTH, HEIGHT);
    
    free(src); free(dst);
    return 0;
}
```

### 📌 파라볼라 변환 (Parabola Transform)

```c
#include <math.h>

// 파라볼라 (감마 보정 유사) - 어두운 영역 강조
void parabolaDark(unsigned char* src, unsigned char* dst, int size) {
    for (int i = 0; i < size; i++) {
        double normalized = src[i] / 255.0;
        double result = normalized * normalized;  // y = x^2
        dst[i] = (unsigned char)(result * 255);
    }
}

// 역 파라볼라 - 밝은 영역 강조
void parabolaBright(unsigned char* src, unsigned char* dst, int size) {
    for (int i = 0; i < size; i++) {
        double normalized = src[i] / 255.0;
        double result = sqrt(normalized);         // y = √x
        dst[i] = (unsigned char)(result * 255);
    }
}

// 파라볼라 (중간값 기준 대칭)
void parabolaSymmetric(unsigned char* src, unsigned char* dst, int size) {
    for (int i = 0; i < size; i++) {
        double x = src[i] / 255.0;
        // y = -4(x - 0.5)^2 + 1 : 중간값이 최대
        double result = -4.0 * (x - 0.5) * (x - 0.5) + 1.0;
        result = (result < 0) ? 0 : (result > 1.0 ? 1.0 : result);
        dst[i] = (unsigned char)(result * 255);
    }
}
```

### 📌 이진화 (Binarization)

```c
// 단순 이진화
void binarize(unsigned char* src, unsigned char* dst, int size, int threshold) {
    for (int i = 0; i < size; i++) {
        dst[i] = (src[i] >= threshold) ? 255 : 0;
    }
}

// 자동 임계값 - 평균값 사용
void binarizeAuto(unsigned char* src, unsigned char* dst, int size) {
    long sum = 0;
    for (int i = 0; i < size; i++) sum += src[i];
    int threshold = (int)(sum / size);
    printf("자동 임계값: %d\n", threshold);
    binarize(src, dst, size, threshold);
}

// 반전 (Inversion)
void invert(unsigned char* src, unsigned char* dst, int size) {
    for (int i = 0; i < size; i++) {
        dst[i] = 255 - src[i];
    }
}

// 감마 보정
void gammaCorrection(unsigned char* src, unsigned char* dst, int size, double gamma) {
    for (int i = 0; i < size; i++) {
        double normalized = src[i] / 255.0;
        double corrected = pow(normalized, 1.0 / gamma);
        dst[i] = (unsigned char)(corrected * 255);
    }
}
```

---

## 1.2 기하학 처리 (Geometric Processing)

기하학 처리는 **이미지의 공간적 위치를 변환**하는 기법입니다.

### 📌 확대 / 축소 (Zoom In / Out)

```c
// 최근접 이웃 보간법 - 확대
void zoomIn(unsigned char* src, unsigned char* dst,
            int srcW, int srcH, int dstW, int dstH) {
    for (int y = 0; y < dstH; y++) {
        for (int x = 0; x < dstW; x++) {
            // 역 매핑: 출력 좌표 → 입력 좌표
            int srcX = (int)(x * srcW / (double)dstW);
            int srcY = (int)(y * srcH / (double)dstH);
            
            // 경계 클리핑
            srcX = (srcX >= srcW) ? srcW - 1 : srcX;
            srcY = (srcY >= srcH) ? srcH - 1 : srcY;
            
            dst[y * dstW + x] = src[srcY * srcW + srcX];
        }
    }
}

// 축소 (평균값 사용)
void zoomOut(unsigned char* src, unsigned char* dst,
             int srcW, int srcH, int dstW, int dstH) {
    double ratioX = (double)srcW / dstW;
    double ratioY = (double)srcH / dstH;
    
    for (int y = 0; y < dstH; y++) {
        for (int x = 0; x < dstW; x++) {
            int srcX = (int)(x * ratioX);
            int srcY = (int)(y * ratioY);
            dst[y * dstW + x] = src[srcY * srcW + srcX];
        }
    }
}

// 바이리니어 보간법 (고품질)
void zoomBilinear(unsigned char* src, unsigned char* dst,
                  int srcW, int srcH, int dstW, int dstH) {
    for (int y = 0; y < dstH; y++) {
        for (int x = 0; x < dstW; x++) {
            double fx = x * (double)(srcW - 1) / (dstW - 1);
            double fy = y * (double)(srcH - 1) / (dstH - 1);
            
            int x0 = (int)fx, y0 = (int)fy;
            int x1 = x0 + 1 < srcW ? x0 + 1 : x0;
            int y1 = y0 + 1 < srcH ? y0 + 1 : y0;
            
            double dx = fx - x0, dy = fy - y0;
            
            double val = (1-dx)*(1-dy) * src[y0*srcW + x0]
                       + dx*(1-dy)     * src[y0*srcW + x1]
                       + (1-dx)*dy     * src[y1*srcW + x0]
                       + dx*dy         * src[y1*srcW + x1];
            
            dst[y * dstW + x] = (unsigned char)val;
        }
    }
}
```

### 📌 회전 (Rotation)

```c
#include <math.h>

#define PI 3.14159265358979

void rotate(unsigned char* src, unsigned char* dst,
            int width, int height, double angle) {
    double rad = angle * PI / 180.0;
    double cosA = cos(rad), sinA = sin(rad);
    double cx = width / 2.0, cy = height / 2.0;
    
    // 배경 초기화 (검정)
    memset(dst, 0, width * height);
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // 역 회전 변환
            double dx = x - cx, dy = y - cy;
            int srcX = (int)(cosA * dx + sinA * dy + cx);
            int srcY = (int)(-sinA * dx + cosA * dy + cy);
            
            if (srcX >= 0 && srcX < width && srcY >= 0 && srcY < height) {
                dst[y * width + x] = src[srcY * width + srcX];
            }
        }
    }
}

// 사용 예
// rotate(src, dst, WIDTH, HEIGHT, 45.0);  // 45도 회전
```

### 📌 좌우/상하 반전 (Flip)

```c
// 좌우 반전
void flipHorizontal(unsigned char* src, unsigned char* dst, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            dst[y * width + x] = src[y * width + (width - 1 - x)];
        }
    }
}

// 상하 반전
void flipVertical(unsigned char* src, unsigned char* dst, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            dst[y * width + x] = src[(height - 1 - y) * width + x];
        }
    }
}

// 이동 (Translation)
void translate(unsigned char* src, unsigned char* dst,
               int width, int height, int tx, int ty) {
    memset(dst, 0, width * height);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int nx = x + tx, ny = y + ty;
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                dst[ny * width + nx] = src[y * width + x];
            }
        }
    }
}
```

---

## 1.3 화소영역 처리 (Area Processing)

화소영역 처리는 **주변 픽셀을 함께 고려**하는 컨볼루션 기반 기법입니다.

```
커널(마스크) 개념:
┌─────┬─────┬─────┐
│ k00 │ k01 │ k02 │
├─────┼─────┼─────┤
│ k10 │ k11 │ k12 │   ← 중심 픽셀 (x, y)
├─────┼─────┼─────┤
│ k20 │ k21 │ k22 │
└─────┴─────┴─────┘
```

### 📌 컨볼루션 함수 (공통)

```c
#include <math.h>

void convolution(unsigned char* src, unsigned char* dst,
                 int width, int height,
                 double kernel[3][3], double divisor) {
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            double sum = 0.0;
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    sum += src[(y + ky) * width + (x + kx)]
                           * kernel[ky + 1][kx + 1];
                }
            }
            sum /= divisor;
            if (sum < 0)   sum = 0;
            if (sum > 255) sum = 255;
            dst[y * width + x] = (unsigned char)sum;
        }
    }
}
```

### 📌 엠보싱 (Embossing)

```c
void emboss(unsigned char* src, unsigned char* dst, int width, int height) {
    // 엠보싱 커널 (부조 효과)
    double kernel[3][3] = {
        {-1, -1,  0},
        {-1,  0,  1},
        { 0,  1,  1}
    };
    
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            double sum = 128.0;  // 회색 기준값 추가
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    sum += src[(y + ky) * width + (x + kx)]
                           * kernel[ky + 1][kx + 1];
                }
            }
            if (sum < 0)   sum = 0;
            if (sum > 255) sum = 255;
            dst[y * width + x] = (unsigned char)sum;
        }
    }
}
```

### 📌 샤프닝 (Sharpening)

```c
void sharpen(unsigned char* src, unsigned char* dst, int width, int height) {
    // 라플라시안 기반 샤프닝
    double kernel[3][3] = {
        { 0, -1,  0},
        {-1,  5, -1},
        { 0, -1,  0}
    };
    convolution(src, dst, width, height, kernel, 1.0);
}

// 강한 샤프닝
void sharpenStrong(unsigned char* src, unsigned char* dst, int width, int height) {
    double kernel[3][3] = {
        {-1, -1, -1},
        {-1,  9, -1},
        {-1, -1, -1}
    };
    convolution(src, dst, width, height, kernel, 1.0);
}
```

### 📌 블러 (Blur)

```c
// 평균 블러 (Mean Blur)
void blurMean(unsigned char* src, unsigned char* dst, int width, int height) {
    double kernel[3][3] = {
        {1, 1, 1},
        {1, 1, 1},
        {1, 1, 1}
    };
    convolution(src, dst, width, height, kernel, 9.0);
}

// 가우시안 블러 (Gaussian Blur)
void blurGaussian(unsigned char* src, unsigned char* dst, int width, int height) {
    double kernel[3][3] = {
        {1, 2, 1},
        {2, 4, 2},
        {1, 2, 1}
    };
    convolution(src, dst, width, height, kernel, 16.0);
}

// 미디언 필터 (Median Filter) - 노이즈 제거
void medianFilter(unsigned char* src, unsigned char* dst, int width, int height) {
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            unsigned char window[9];
            int k = 0;
            for (int ky = -1; ky <= 1; ky++)
                for (int kx = -1; kx <= 1; kx++)
                    window[k++] = src[(y + ky) * width + (x + kx)];
            
            // 버블 정렬
            for (int i = 0; i < 8; i++)
                for (int j = 0; j < 8 - i; j++)
                    if (window[j] > window[j + 1]) {
                        unsigned char tmp = window[j];
                        window[j] = window[j + 1];
                        window[j + 1] = tmp;
                    }
            dst[y * width + x] = window[4];  // 중앙값
        }
    }
}
```

### 📌 엣지 검출 (Edge Detection)

```c
// 소벨 엣지 (Sobel Edge)
void sobelEdge(unsigned char* src, unsigned char* dst, int width, int height) {
    double kernelX[3][3] = {{ -1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    double kernelY[3][3] = {{ -1,-2,-1}, { 0, 0, 0}, { 1, 2, 1}};
    
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            double gx = 0, gy = 0;
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    double p = src[(y + ky) * width + (x + kx)];
                    gx += p * kernelX[ky + 1][kx + 1];
                    gy += p * kernelY[ky + 1][kx + 1];
                }
            }
            double mag = sqrt(gx * gx + gy * gy);
            dst[y * width + x] = (mag > 255) ? 255 : (unsigned char)mag;
        }
    }
}
```

---

# 2. Python을 이용한 영상처리 (RAW + RGB)

## 환경 설정

```python
import numpy as np
import matplotlib.pyplot as plt
import struct
import os

# RAW 이미지 읽기 / 저장
def read_raw(filename, width=256, height=256):
    with open(filename, 'rb') as f:
        data = f.read()
    return np.frombuffer(data, dtype=np.uint8).reshape((height, width))

def write_raw(filename, img):
    with open(filename, 'wb') as f:
        f.write(img.astype(np.uint8).tobytes())

# 시각화 유틸리티
def show_images(images, titles, cols=3, cmap='gray'):
    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes
    for i, (img, title) in enumerate(zip(images, titles)):
        axes[i].imshow(img, cmap=cmap)
        axes[i].set_title(title)
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()
```

---

## 2.1 화소점 처리

```python
import numpy as np

class PixelPointProcessing:
    """화소점 처리 클래스"""
    
    @staticmethod
    def brighten(img, value=50):
        """밝기 증가"""
        return np.clip(img.astype(int) + value, 0, 255).astype(np.uint8)
    
    @staticmethod
    def darken(img, value=50):
        """밝기 감소"""
        return np.clip(img.astype(int) - value, 0, 255).astype(np.uint8)
    
    @staticmethod
    def parabola_dark(img):
        """파라볼라 - 어두운 영역 강조 (y = x²)"""
        normalized = img / 255.0
        result = normalized ** 2
        return (result * 255).astype(np.uint8)
    
    @staticmethod
    def parabola_bright(img):
        """파라볼라 - 밝은 영역 강조 (y = √x)"""
        normalized = img / 255.0
        result = np.sqrt(normalized)
        return (result * 255).astype(np.uint8)
    
    @staticmethod
    def parabola_symmetric(img):
        """대칭 파라볼라 - 중간값 강조 (y = -4(x-0.5)² + 1)"""
        x = img / 255.0
        result = np.clip(-4.0 * (x - 0.5) ** 2 + 1.0, 0, 1)
        return (result * 255).astype(np.uint8)
    
    @staticmethod
    def binarize(img, threshold=128):
        """이진화"""
        return np.where(img >= threshold, 255, 0).astype(np.uint8)
    
    @staticmethod
    def binarize_auto(img):
        """자동 이진화 (평균값 기준)"""
        threshold = int(img.mean())
        print(f"자동 임계값: {threshold}")
        return np.where(img >= threshold, 255, 0).astype(np.uint8)
    
    @staticmethod
    def invert(img):
        """반전"""
        return (255 - img).astype(np.uint8)
    
    @staticmethod
    def gamma_correction(img, gamma=2.2):
        """감마 보정"""
        normalized = img / 255.0
        corrected = normalized ** (1.0 / gamma)
        return (corrected * 255).astype(np.uint8)
    
    @staticmethod
    def contrast_stretch(img):
        """대비 스트레칭 (히스토그램 정규화)"""
        min_val, max_val = img.min(), img.max()
        if max_val == min_val:
            return img.copy()
        return ((img - min_val) * 255.0 / (max_val - min_val)).astype(np.uint8)
    
    @staticmethod
    def histogram_equalization(img):
        """히스토그램 평활화"""
        hist, bins = np.histogram(img.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
        lut = np.uint8(cdf_normalized)
        return lut[img]

# ── 실행 예시 ──────────────────────────────────────────
if __name__ == "__main__":
    img = read_raw("input.raw")
    ppp = PixelPointProcessing
    
    results = [img,
               ppp.brighten(img, 50),
               ppp.parabola_bright(img),
               ppp.parabola_dark(img),
               ppp.binarize(img, 128),
               ppp.histogram_equalization(img)]
    titles = ["원본", "밝게(+50)", "파라볼라(밝게)",
              "파라볼라(어둡게)", "이진화(128)", "히스토그램 평활화"]
    
    show_images(results, titles)
```

---

## 2.2 기하학 처리

```python
from scipy.ndimage import zoom as scipy_zoom
import cv2  # optional

class GeometricProcessing:
    """기하학 처리 클래스"""
    
    @staticmethod
    def zoom_in_nearest(img, scale=2.0):
        """확대 - 최근접 이웃 보간"""
        h, w = img.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        y_idx = (np.arange(new_h) / scale).astype(int).clip(0, h - 1)
        x_idx = (np.arange(new_w) / scale).astype(int).clip(0, w - 1)
        return img[np.ix_(y_idx, x_idx)]
    
    @staticmethod
    def zoom_in_bilinear(img, scale=2.0):
        """확대 - 바이리니어 보간"""
        h, w = img.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        
        y = np.linspace(0, h - 1, new_h)
        x = np.linspace(0, w - 1, new_w)
        
        y0 = np.floor(y).astype(int).clip(0, h - 2)
        x0 = np.floor(x).astype(int).clip(0, w - 2)
        y1, x1 = y0 + 1, x0 + 1
        
        dy = (y - y0)[:, np.newaxis]
        dx = (x - x0)[np.newaxis, :]
        
        result = ((1 - dy) * (1 - dx) * img[y0[:, None], x0[None, :]]
                + (1 - dy) * dx       * img[y0[:, None], x1[None, :]]
                +  dy      * (1 - dx) * img[y1[:, None], x0[None, :]]
                +  dy      * dx       * img[y1[:, None], x1[None, :]])
        return result.astype(np.uint8)
    
    @staticmethod
    def zoom_out(img, scale=0.5):
        """축소"""
        h, w = img.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        y_idx = (np.arange(new_h) / scale).astype(int).clip(0, h - 1)
        x_idx = (np.arange(new_w) / scale).astype(int).clip(0, w - 1)
        return img[np.ix_(y_idx, x_idx)]
    
    @staticmethod
    def rotate(img, angle):
        """회전 (역 매핑)"""
        h, w = img.shape[:2]
        rad = np.radians(angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        cx, cy = w / 2, h / 2
        
        dst = np.zeros_like(img)
        for y in range(h):
            for x in range(w):
                dx, dy = x - cx, y - cy
                src_x = int(cos_a * dx + sin_a * dy + cx)
                src_y = int(-sin_a * dx + cos_a * dy + cy)
                if 0 <= src_x < w and 0 <= src_y < h:
                    dst[y, x] = img[src_y, src_x]
        return dst
    
    @staticmethod
    def rotate_fast(img, angle):
        """회전 - numpy 벡터화 (빠름)"""
        h, w = img.shape[:2]
        rad = np.radians(angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        cx, cy = w / 2, h / 2
        
        y_idx, x_idx = np.mgrid[0:h, 0:w]
        dx = x_idx - cx
        dy = y_idx - cy
        
        src_x = (cos_a * dx + sin_a * dy + cx).astype(int)
        src_y = (-sin_a * dx + cos_a * dy + cy).astype(int)
        
        mask = (src_x >= 0) & (src_x < w) & (src_y >= 0) & (src_y < h)
        dst = np.zeros_like(img)
        dst[mask] = img[src_y[mask], src_x[mask]]
        return dst
    
    @staticmethod
    def flip_horizontal(img):
        return img[:, ::-1]
    
    @staticmethod
    def flip_vertical(img):
        return img[::-1, :]
    
    @staticmethod
    def translate(img, tx=30, ty=20):
        """이동 변환"""
        h, w = img.shape[:2]
        dst = np.zeros_like(img)
        for y in range(h):
            for x in range(w):
                nx, ny = x + tx, y + ty
                if 0 <= nx < w and 0 <= ny < h:
                    dst[ny, nx] = img[y, x]
        return dst

# ── 실행 예시 ──────────────────────────────────────────
if __name__ == "__main__":
    img = read_raw("input.raw")
    gp = GeometricProcessing
    
    results = [img,
               gp.zoom_in_nearest(img, 2.0),
               gp.zoom_in_bilinear(img, 2.0),
               gp.zoom_out(img, 0.5),
               gp.rotate_fast(img, 45),
               gp.flip_horizontal(img)]
    titles = ["원본", "확대(x2, 최근접)", "확대(x2, 바이리니어)",
              "축소(x0.5)", "회전(45°)", "좌우반전"]
    
    show_images(results, titles)
```

### 📌 RGB 이미지 기하학 처리

```python
import matplotlib.pyplot as plt

def read_rgb(filename, width=256, height=256):
    """RGB RAW 파일 읽기"""
    with open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape((height, width, 3))

def rotate_rgb(img, angle):
    """RGB 이미지 회전"""
    r = GeometricProcessing.rotate_fast(img[:,:,0], angle)
    g = GeometricProcessing.rotate_fast(img[:,:,1], angle)
    b = GeometricProcessing.rotate_fast(img[:,:,2], angle)
    return np.stack([r, g, b], axis=2)
```

---

## 2.3 화소영역 처리

```python
from scipy.ndimage import convolve
import numpy as np

class AreaProcessing:
    """화소영역 처리 클래스"""
    
    @staticmethod
    def _convolve(img, kernel):
        """컨볼루션 공통 함수"""
        result = convolve(img.astype(float), kernel)
        return np.clip(result, 0, 255).astype(np.uint8)
    
    @staticmethod
    def emboss(img):
        """엠보싱"""
        kernel = np.array([[-1, -1, 0],
                           [-1,  0, 1],
                           [ 0,  1, 1]], dtype=float)
        result = convolve(img.astype(float), kernel) + 128
        return np.clip(result, 0, 255).astype(np.uint8)
    
    @staticmethod
    def sharpen(img):
        """샤프닝"""
        kernel = np.array([[ 0, -1,  0],
                           [-1,  5, -1],
                           [ 0, -1,  0]], dtype=float)
        return AreaProcessing._convolve(img, kernel)
    
    @staticmethod
    def sharpen_strong(img):
        """강한 샤프닝"""
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]], dtype=float)
        return AreaProcessing._convolve(img, kernel)
    
    @staticmethod
    def blur_mean(img):
        """평균 블러"""
        kernel = np.ones((3, 3), dtype=float) / 9
        return AreaProcessing._convolve(img, kernel)
    
    @staticmethod
    def blur_gaussian(img):
        """가우시안 블러"""
        kernel = np.array([[1, 2, 1],
                           [2, 4, 2],
                           [1, 2, 1]], dtype=float) / 16
        return AreaProcessing._convolve(img, kernel)
    
    @staticmethod
    def blur_gaussian_5x5(img):
        """5×5 가우시안 블러"""
        kernel = np.array([
            [ 1,  4,  7,  4, 1],
            [ 4, 16, 26, 16, 4],
            [ 7, 26, 41, 26, 7],
            [ 4, 16, 26, 16, 4],
            [ 1,  4,  7,  4, 1]], dtype=float) / 273
        return AreaProcessing._convolve(img, kernel)
    
    @staticmethod
    def median_filter(img, size=3):
        """미디언 필터"""
        from scipy.ndimage import median_filter
        return median_filter(img, size=size)
    
    @staticmethod
    def edge_sobel(img):
        """소벨 엣지 검출"""
        kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
        ky = np.array([[-1,-2,-1], [ 0, 0, 0], [ 1, 2, 1]], dtype=float)
        gx = convolve(img.astype(float), kx)
        gy = convolve(img.astype(float), ky)
        mag = np.sqrt(gx**2 + gy**2)
        return np.clip(mag, 0, 255).astype(np.uint8)
    
    @staticmethod
    def edge_laplacian(img):
        """라플라시안 엣지 검출"""
        kernel = np.array([[ 0, -1,  0],
                           [-1,  4, -1],
                           [ 0, -1,  0]], dtype=float)
        return AreaProcessing._convolve(img, kernel)

# ── 실행 예시 ──────────────────────────────────────────
if __name__ == "__main__":
    img = read_raw("input.raw")
    ap = AreaProcessing
    
    results = [img, ap.emboss(img), ap.sharpen(img),
               ap.blur_mean(img), ap.blur_gaussian(img), ap.edge_sobel(img)]
    titles = ["원본", "엠보싱", "샤프닝",
              "평균 블러", "가우시안 블러", "소벨 엣지"]
    show_images(results, titles)
```

---

# 3. Python + Pillow + 마우스 처리

> 마우스로 드래그하여 사각형 ROI(Region of Interest)를 선택하고,  
> 해당 영역에만 화소처리를 적용합니다.

```python
import tkinter as tk
from PIL import Image, ImageTk, ImageFilter, ImageEnhance
import numpy as np

class ROIImageProcessor:
    """마우스 ROI 선택 + 영상처리 애플리케이션"""
    
    def __init__(self, image_path):
        self.root = tk.Tk()
        self.root.title("ROI 영상처리 - 마우스로 영역 선택")
        
        # 이미지 로드
        self.original = Image.open(image_path).convert("RGB")
        self.current   = self.original.copy()
        self.display   = self.original.copy()
        
        self.start_x = self.start_y = 0
        self.rect_id = None
        self.roi     = None
        
        self._build_ui()
    
    def _build_ui(self):
        # ── 툴바 ──
        toolbar = tk.Frame(self.root, bg="#2c2c2c", pady=4)
        toolbar.pack(fill=tk.X)
        
        ops = [
            ("밝게",       self.apply_brighten),
            ("어둡게",     self.apply_darken),
            ("이진화",     self.apply_binarize),
            ("샤프닝",     self.apply_sharpen),
            ("블러",       self.apply_blur),
            ("엠보싱",     self.apply_emboss),
            ("엣지",       self.apply_edge),
            ("확대(x2)",   self.apply_zoom),
            ("회전(30°)",  self.apply_rotate),
            ("원본복원",   self.reset_image),
        ]
        for label, cmd in ops:
            tk.Button(toolbar, text=label, command=cmd,
                      bg="#444", fg="white", relief=tk.FLAT,
                      padx=8, pady=2).pack(side=tk.LEFT, padx=2)
        
        # ── 캔버스 ──
        self.canvas = tk.Canvas(self.root,
                                width=self.original.width,
                                height=self.original.height,
                                cursor="crosshair")
        self.canvas.pack()
        
        # 마우스 이벤트
        self.canvas.bind("<ButtonPress-1>",   self.on_mouse_press)
        self.canvas.bind("<B1-Motion>",       self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_release)
        
        # 상태바
        self.status = tk.StringVar(value="마우스를 드래그하여 처리할 영역을 선택하세요.")
        tk.Label(self.root, textvariable=self.status,
                 bg="#1e1e1e", fg="#aaa", anchor="w").pack(fill=tk.X)
        
        self._refresh()
    
    def _refresh(self):
        self.tk_img = ImageTk.PhotoImage(self.display)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)
    
    # ── 마우스 이벤트 핸들러 ──────────────────────────
    def on_mouse_press(self, event):
        self.start_x, self.start_y = event.x, event.y
        if self.rect_id:
            self.canvas.delete(self.rect_id)
    
    def on_mouse_drag(self, event):
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        self.rect_id = self.canvas.create_rectangle(
            self.start_x, self.start_y, event.x, event.y,
            outline="#00FF88", width=2, dash=(4, 4))
    
    def on_mouse_release(self, event):
        x1 = min(self.start_x, event.x)
        y1 = min(self.start_y, event.y)
        x2 = max(self.start_x, event.x)
        y2 = max(self.start_y, event.y)
        self.roi = (x1, y1, x2, y2)
        self.status.set(f"선택 영역: ({x1},{y1}) ~ ({x2},{y2})  → 위의 버튼으로 처리")
    
    # ── 처리 공통 로직 ────────────────────────────────
    def _apply_to_roi(self, func):
        if not self.roi:
            self.status.set("⚠  먼저 마우스로 영역을 선택하세요!")
            return
        x1, y1, x2, y2 = self.roi
        region = self.current.crop((x1, y1, x2, y2))
        processed = func(region)
        result = self.current.copy()
        result.paste(processed, (x1, y1))
        self.current = result
        self.display = self.current.copy()
        
        # ROI 테두리 표시
        self._refresh()
        self.canvas.create_rectangle(x1, y1, x2, y2,
                                     outline="#FF4444", width=1)
    
    # ── 처리 함수들 ───────────────────────────────────
    def apply_brighten(self):
        self._apply_to_roi(lambda img: ImageEnhance.Brightness(img).enhance(1.8))
        self.status.set("✔  ROI 밝기 증가 완료")
    
    def apply_darken(self):
        self._apply_to_roi(lambda img: ImageEnhance.Brightness(img).enhance(0.4))
        self.status.set("✔  ROI 밝기 감소 완료")
    
    def apply_binarize(self):
        def binarize(img):
            gray = img.convert("L")
            bw   = gray.point(lambda p: 255 if p >= 128 else 0)
            return bw.convert("RGB")
        self._apply_to_roi(binarize)
        self.status.set("✔  ROI 이진화 완료")
    
    def apply_sharpen(self):
        self._apply_to_roi(lambda img: img.filter(ImageFilter.SHARPEN))
        self.status.set("✔  ROI 샤프닝 완료")
    
    def apply_blur(self):
        self._apply_to_roi(lambda img: img.filter(ImageFilter.GaussianBlur(radius=3)))
        self.status.set("✔  ROI 가우시안 블러 완료")
    
    def apply_emboss(self):
        self._apply_to_roi(lambda img: img.filter(ImageFilter.EMBOSS))
        self.status.set("✔  ROI 엠보싱 완료")
    
    def apply_edge(self):
        self._apply_to_roi(lambda img: img.filter(ImageFilter.FIND_EDGES))
        self.status.set("✔  ROI 엣지 검출 완료")
    
    def apply_zoom(self):
        def zoom(img):
            w, h = img.size
            zoomed = img.resize((w * 2, h * 2), Image.BILINEAR)
            return zoomed.crop((0, 0, w, h))  # ROI 크기 유지
        self._apply_to_roi(zoom)
        self.status.set("✔  ROI 확대(x2) 완료")
    
    def apply_rotate(self):
        self._apply_to_roi(lambda img: img.rotate(30, expand=False))
        self.status.set("✔  ROI 회전(30°) 완료")
    
    def reset_image(self):
        self.current = self.original.copy()
        self.display = self.original.copy()
        self.roi     = None
        self._refresh()
        self.status.set("원본 이미지로 복원되었습니다.")
    
    def run(self):
        self.root.mainloop()

# ── 실행 ──────────────────────────────────────────────
if __name__ == "__main__":
    app = ROIImageProcessor("sample.png")   # PNG, JPG, BMP 모두 가능
    app.run()
```

**사용 방법:**
1. 캔버스 위에서 마우스 드래그로 영역 선택
2. 상단 버튼을 클릭하여 선택 영역에만 처리 적용
3. `원본복원` 버튼으로 언제든지 리셋 가능

---

# 4. Python + scikit-learn 머신러닝

## 4.1 붓꽃 종류 감별 (Iris Classification)

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def iris_classification():
    """붓꽃 분류 - 여러 알고리즘 비교"""
    
    # ① 데이터 로드
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = iris.feature_names
    class_names   = iris.target_names
    
    print("=" * 50)
    print("붓꽃 데이터셋 정보")
    print("=" * 50)
    print(f"샘플 수    : {X.shape[0]}")
    print(f"특징 수    : {X.shape[1]}")
    print(f"클래스     : {list(class_names)}")
    print(f"특징 이름  : {feature_names}")
    
    # ② 데이터 분할 + 정규화
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    
    # ③ 모델 학습 & 비교
    models = {
        "KNN (k=5)"       : KNeighborsClassifier(n_neighbors=5),
        "SVM (RBF)"       : SVC(kernel='rbf', C=1.0, random_state=42),
        "Decision Tree"   : DecisionTreeClassifier(max_depth=4, random_state=42),
        "Random Forest"   : RandomForestClassifier(n_estimators=100, random_state=42),
    }
    
    best_model, best_acc = None, 0
    for name, model in models.items():
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        print(f"\n[{name}]  정확도: {acc:.4f}")
        print(classification_report(y_test, model.predict(X_test),
                                    target_names=class_names))
        if acc > best_acc:
            best_acc, best_model = acc, model
    
    # ④ 혼동 행렬 시각화
    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'혼동 행렬 (최고 정확도: {best_acc:.4f})')
    plt.ylabel('실제 클래스')
    plt.xlabel('예측 클래스')
    plt.tight_layout()
    plt.show()
    
    # ⑤ 새 데이터 예측
    new_flower = np.array([[5.1, 3.5, 1.4, 0.2]])
    new_scaled = scaler.transform(new_flower)
    pred = best_model.predict(new_scaled)
    print(f"\n새 데이터 예측: {class_names[pred[0]]}")

iris_classification()
```

---

## 4.2 와인 등급 감별 (Wine Quality Classification)

```python
from sklearn.datasets import load_wine
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd

def wine_classification():
    """와인 등급 분류"""
    
    # ① 데이터 로드
    wine = load_wine()
    X, y  = wine.data, wine.target
    df    = pd.DataFrame(X, columns=wine.feature_names)
    df['class'] = y
    
    print("=" * 50)
    print("와인 데이터셋 정보")
    print("=" * 50)
    print(df.describe())
    print(f"\n클래스 분포:\n{df['class'].value_counts()}")
    
    # ② 전처리 + 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    
    # ③ 그레디언트 부스팅
    model = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05,
        max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, model.predict(X_test))
    cv_scores = cross_val_score(model, X, y, cv=5)
    
    print(f"\n테스트 정확도   : {acc:.4f}")
    print(f"교차 검증 평균  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print("\n분류 보고서:")
    print(classification_report(y_test, model.predict(X_test),
                                 target_names=wine.target_names))
    
    # ④ 특징 중요도 시각화
    fi = pd.Series(model.feature_importances_, index=wine.feature_names)
    fi_sorted = fi.sort_values(ascending=False)
    
    plt.figure(figsize=(10, 5))
    fi_sorted.plot(kind='bar', color='steelblue')
    plt.title("와인 품질 예측 - 특징 중요도")
    plt.xlabel("특징")
    plt.ylabel("중요도")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

wine_classification()
```

---

## 4.3 카드 사기 감별 (Credit Card Fraud Detection)

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, roc_curve
import pandas as pd
import numpy as np

def fraud_detection_demo():
    """
    신용카드 사기 감지 - 불균형 데이터 처리
    (실제 데이터셋 대신 시뮬레이션 데이터 사용)
    """
    np.random.seed(42)
    n_normal = 9700
    n_fraud  = 300
    
    # ── 시뮬레이션 데이터 생성 ──
    normal = np.random.multivariate_normal(
        mean=[0] * 10, cov=np.eye(10) * 2, size=n_normal)
    fraud  = np.random.multivariate_normal(
        mean=[3, -2, 1, 4, -3, 2, -1, 3, -4, 2], cov=np.eye(10), size=n_fraud)
    
    X = np.vstack([normal, fraud])
    y = np.hstack([np.zeros(n_normal), np.ones(n_fraud)])
    
    cols = [f'V{i+1}' for i in range(10)]
    df = pd.DataFrame(X, columns=cols)
    df['Amount'] = np.abs(np.random.exponential(scale=100, size=len(y)))
    df['Class']  = y
    
    print("=" * 50)
    print("카드 사기 감지 데이터셋")
    print("=" * 50)
    print(f"정상 거래: {n_normal} ({n_normal/(n_normal+n_fraud)*100:.1f}%)")
    print(f"사기 거래: {n_fraud}  ({n_fraud/(n_normal+n_fraud)*100:.1f}%)")
    
    # ── 전처리 ──
    X_data = df.drop('Class', axis=1).values
    y_data = df['Class'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.2, random_state=42, stratify=y_data)
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    
    # ── 모델 1: 로지스틱 회귀 (class_weight 조정) ──
    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    y_proba_lr = lr.predict_proba(X_test)[:, 1]
    auc_lr = roc_auc_score(y_test, y_proba_lr)
    
    # ── 모델 2: Isolation Forest (비지도) ──
    iso = IsolationForest(contamination=0.03, random_state=42)
    iso.fit(X_train)
    y_pred_iso = (iso.predict(X_test) == -1).astype(int)
    
    print(f"\n[로지스틱 회귀]  AUC-ROC: {auc_lr:.4f}")
    print(classification_report(y_test, (y_proba_lr > 0.5).astype(int),
                                 target_names=['정상', '사기']))
    print(f"\n[Isolation Forest]")
    print(classification_report(y_test, y_pred_iso,
                                 target_names=['정상', '사기']))
    
    # ── ROC 커브 ──
    fpr, tpr, _ = roc_curve(y_test, y_proba_lr)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'로지스틱 회귀 (AUC = {auc_lr:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('카드 사기 감지 - ROC 커브')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

fraud_detection_demo()
```

---

# 5. Python + OpenCV 영상처리 및 검출

## 설치

```bash
pip install opencv-python opencv-contrib-python numpy matplotlib
```

## 5.1 영상처리

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_cv(images, titles, cols=3):
    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = np.array(axes).flatten()
    for i, (img, title) in enumerate(zip(images, titles)):
        disp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else img
        axes[i].imshow(disp, cmap=None if img.ndim == 3 else 'gray')
        axes[i].set_title(title)
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

# ── 이미지 로드 ──
img_color = cv2.imread("sample.jpg")
img_gray  = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

# ══ 화소점 처리 ═══════════════════════════════════════
def pixel_point_demo(img, gray):
    # 밝기 조절
    bright = cv2.convertScaleAbs(img, alpha=1.5, beta=30)
    dark   = cv2.convertScaleAbs(img, alpha=0.5, beta=-30)
    
    # 이진화
    _, binary    = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    _, otsu      = cv2.threshold(gray, 0,   255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive_bin = cv2.adaptiveThreshold(gray, 255,
                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                       cv2.THRESH_BINARY, 11, 2)
    
    # 반전
    inverted = cv2.bitwise_not(img)
    
    # 감마 보정
    gamma = 2.2
    lut   = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                       for i in range(256)], dtype=np.uint8)
    gamma_corrected = cv2.LUT(img, lut)
    
    # 히스토그램 평활화
    hist_eq = cv2.equalizeHist(gray)
    
    return (bright, dark, binary, otsu, adaptive_bin, inverted)


# ══ 기하학 처리 ═══════════════════════════════════════
def geometric_demo(img, gray):
    h, w = img.shape[:2]
    
    # 확대 / 축소
    zoom_in  = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)
    zoom_out = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    
    # 회전
    M_rot = cv2.getRotationMatrix2D((w // 2, h // 2), 45, 1.0)
    rotated = cv2.warpAffine(img, M_rot, (w, h))
    
    # 아핀 변환 (기울이기)
    pts1 = np.float32([[0, 0], [w, 0], [0, h]])
    pts2 = np.float32([[w*0.1, h*0.1], [w*0.9, h*0.05], [w*0.2, h*0.9]])
    M_aff = cv2.getAffineTransform(pts1, pts2)
    affine = cv2.warpAffine(img, M_aff, (w, h))
    
    # 원근 변환
    pts1 = np.float32([[50, 50], [w-50, 50], [50, h-50], [w-50, h-50]])
    pts2 = np.float32([[0, 0],   [w, 0],     [0, h],     [w, h]])
    M_per = cv2.getPerspectiveTransform(pts1, pts2)
    perspective = cv2.warpPerspective(img, M_per, (w, h))
    
    # 좌우 반전
    flipped = cv2.flip(img, 1)
    
    return (zoom_in, zoom_out, rotated, affine, perspective, flipped)


# ══ 화소영역 처리 ═════════════════════════════════════
def area_processing_demo(img, gray):
    # 블러
    blur_avg    = cv2.blur(img, (5, 5))
    blur_gauss  = cv2.GaussianBlur(img, (5, 5), 0)
    blur_median = cv2.medianBlur(img, 5)
    blur_bi     = cv2.bilateralFilter(img, 9, 75, 75)
    
    # 샤프닝
    kernel_sharp = np.array([[ 0, -1,  0],
                              [-1,  5, -1],
                              [ 0, -1,  0]])
    sharpened = cv2.filter2D(img, -1, kernel_sharp)
    
    # 엠보싱
    kernel_emboss = np.array([[-2, -1, 0],
                               [-1,  1, 1],
                               [ 0,  1, 2]])
    embossed = cv2.filter2D(gray, -1, kernel_emboss) + 128
    
    # 엣지 검출
    edges_canny  = cv2.Canny(gray, 100, 200)
    edges_sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    edges_sobel_x = np.abs(edges_sobel_x).astype(np.uint8)
    edges_laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    edges_laplacian = np.abs(edges_laplacian).astype(np.uint8)
    
    # 모폴로지
    kernel_morph = np.ones((5, 5), np.uint8)
    dilated  = cv2.dilate(gray, kernel_morph, iterations=1)
    eroded   = cv2.erode(gray, kernel_morph, iterations=1)
    opened   = cv2.morphologyEx(gray, cv2.MORPH_OPEN,  kernel_morph)
    closed   = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel_morph)
    gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel_morph)
    
    show_cv([blur_avg, blur_gauss, blur_median, blur_bi,
             sharpened, embossed, edges_canny, edges_sobel_x],
            ["평균 블러", "가우시안 블러", "미디언 블러", "양방향 필터",
             "샤프닝", "엠보싱", "캐니 엣지", "소벨 X 엣지"])
    
    show_cv([gray, dilated, eroded, opened, closed, gradient],
            ["원본(그레이)", "팽창", "침식", "오프닝", "클로징", "모폴로지 그래디언트"])
    
    return True

# ── 실행 ──
pixel_point_demo(img_color, img_gray)
geometric_demo(img_color, img_gray)
area_processing_demo(img_color, img_gray)
```

---

## 5.2 이미지 검출 (Haar Cascade Detection)

```python
import cv2
import os

class FaceDetector:
    """Haar Cascade를 이용한 다중 객체 검출"""
    
    CASCADE_FILES = {
        "얼굴"     : "haarcascade_frontalface_default.xml",
        "고양이얼굴": "haarcascade_frontalcatface.xml",
        "눈"       : "haarcascade_eye.xml",
        "상반신"   : "haarcascade_upperbody.xml",
        "정면얼굴2": "haarcascade_frontalface_alt2.xml",
    }
    
    COLORS = {
        "얼굴"     : (0, 255, 0),
        "고양이얼굴": (255, 165, 0),
        "눈"       : (255, 0, 0),
        "상반신"   : (0, 0, 255),
        "정면얼굴2": (0, 255, 255),
    }
    
    def __init__(self):
        self.cascades = {}
        cascade_base = cv2.data.haarcascades
        
        for name, filename in self.CASCADE_FILES.items():
            path = os.path.join(cascade_base, filename)
            if os.path.exists(path):
                self.cascades[name] = cv2.CascadeClassifier(path)
                print(f"✔ Cascade 로드: {name}")
            else:
                print(f"✘ 파일 없음: {filename}")
    
    def detect_image(self, img_path, target="얼굴"):
        """정지 이미지에서 검출"""
        img  = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)  # 대비 향상
        
        if target not in self.cascades:
            print(f"'{target}' 캐스케이드가 없습니다.")
            return
        
        cascade = self.cascades[target]
        color   = self.COLORS.get(target, (0, 255, 0))
        
        objects = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        print(f"검출된 {target} 수: {len(objects)}")
        for (x, y, w, h) in objects:
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(img, target, (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.imshow(f"{target} 검출 결과", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def detect_realtime(self, target="얼굴"):
        """실시간 카메라 검출"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("카메라를 열 수 없습니다.")
            return
        
        cascade = self.cascades.get(target)
        color   = self.COLORS.get(target, (0, 255, 0))
        
        print(f"실시간 {target} 검출 시작 (q 키로 종료)")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            
            objects = cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            for (x, y, w, h) in objects:
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                label = f"{target} ({w}x{h})"
                cv2.putText(frame, label, (x, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # 눈 검출 (얼굴 ROI 내부에서)
                if target == "얼굴" and "눈" in self.cascades:
                    roi_gray = gray[y:y+h, x:x+w]
                    roi_color = frame[y:y+h, x:x+w]
                    eyes = self.cascades["눈"].detectMultiScale(
                        roi_gray, scaleFactor=1.1, minNeighbors=10)
                    for (ex, ey, ew, eh) in eyes:
                        cv2.rectangle(roi_color, (ex, ey),
                                      (ex+ew, ey+eh), (255, 0, 0), 1)
            
            # FPS 표시
            cv2.putText(frame, f"Target: {target} | Press 'q' to quit",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2)
            
            cv2.imshow(f"실시간 {target} 검출", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def detect_multi(self, img_path):
        """여러 객체 동시 검출"""
        img  = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        total = 0
        for name, cascade in self.cascades.items():
            color   = self.COLORS.get(name, (255, 255, 255))
            objects = cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            for (x, y, w, h) in objects:
                cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                cv2.putText(img, name, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            total += len(objects)
            if len(objects) > 0:
                print(f"  {name}: {len(objects)}개 검출")
        
        print(f"총 검출: {total}개")
        cv2.imshow("다중 객체 검출", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ── 실행 예시 ──────────────────────────────────────────
if __name__ == "__main__":
    detector = FaceDetector()
    
    # 1. 정지 이미지에서 얼굴 검출
    detector.detect_image("people.jpg", target="얼굴")
    
    # 2. 실시간 카메라 검출
    detector.detect_realtime(target="얼굴")
    
    # 3. 다중 객체 동시 검출
    detector.detect_multi("group_photo.jpg")
```

---

## 📊 처리 기법 요약표

| 분류 | 기법 | C/C++ | Python | Pillow | OpenCV |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **화소점** | 밝기 조절 | ✅ | ✅ | ✅ | ✅ |
| **화소점** | 파라볼라 | ✅ | ✅ | ✅ | ✅ |
| **화소점** | 이진화 | ✅ | ✅ | ✅ | ✅ (Otsu 포함) |
| **화소점** | 히스토그램 평활화 | ✅ | ✅ | - | ✅ |
| **기하학** | 확대/축소 | ✅ | ✅ | ✅ | ✅ |
| **기하학** | 회전 | ✅ | ✅ | ✅ | ✅ (아핀) |
| **기하학** | 원근 변환 | - | - | - | ✅ |
| **영역** | 블러 | ✅ | ✅ | ✅ | ✅ (양방향 포함) |
| **영역** | 샤프닝 | ✅ | ✅ | ✅ | ✅ |
| **영역** | 엠보싱 | ✅ | ✅ | ✅ | ✅ |
| **영역** | 엣지 검출 | ✅ (소벨) | ✅ | - | ✅ (캐니/소벨) |
| **영역** | 모폴로지 | - | - | - | ✅ |
| **검출** | Haar Cascade | - | - | - | ✅ |
| **ML** | 분류 | - | ✅ (sklearn) | - | - |

---

## 📚 참고 자료

- [OpenCV 공식 문서](https://docs.opencv.org/)
- [scikit-learn 공식 문서](https://scikit-learn.org/stable/)
- [Pillow 공식 문서](https://pillow.readthedocs.io/)
- [NumPy 공식 문서](https://numpy.org/doc/)
- [OpenCV Haar Cascade 모델 목록](https://github.com/opencv/opencv/tree/master/data/haarcascades)

---

## 📄 라이선스

본 교육 자료는 **MIT License** 하에 자유롭게 사용 및 배포할 수 있습니다.

---

> 📧 문의 및 기여: Issues 또는 Pull Request를 통해 참여해 주세요.
