#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "cmath"

using namespace std;
using namespace cv;


//좌우 변환 함수
void LRreverse(Mat mat) {

   uchar* data = mat.data;

   for (int row = 0; row < mat.rows; row++)
   {
       for (int col = 0; col < mat.cols / 2; col++)
       {
           int matCols = mat.cols - col - 1;

           uchar b = data[row * mat.cols * 3 + col * 3];
           uchar g = data[row * mat.cols * 3 + col * 3 + 1];
           uchar r = data[row * mat.cols * 3 + col * 3 + 2];

           data[row * mat.cols * 3 + col * 3] = data[row * mat.cols * 3 + matCols * 3];
           data[row * mat.cols * 3 + col * 3 + 1] = data[row * mat.cols * 3 + matCols * 3 + 1];
           data[row * mat.cols * 3 + col * 3 + 2] = data[row * mat.cols * 3 + matCols * 3 + 2];

           data[row * mat.cols * 3 + matCols * 3] = b;
           data[row * mat.cols * 3 + matCols * 3 + 1] = g;
           data[row * mat.cols * 3 + matCols * 3 + 2] = r;

       }
   }

}


void SetGray(Mat mat) {

   uchar* data = mat.data;

   for (int row = 0; row < mat.rows; row++)
   {
       for (int col = 0; col < mat.cols; col++)
       {
           uchar b = data[row * mat.cols * 3 + col * 3];
           uchar g = data[row * mat.cols * 3 + col * 3 + 1];
           uchar r = data[row * mat.cols * 3 + col * 3 + 2];

           uchar gray = (b + g + r) / 3;

           data[row * mat.cols * 3 + col * 3] = gray;
           data[row * mat.cols * 3 + col * 3 + 1] = gray;
           data[row * mat.cols * 3 + col * 3 + 2] = gray;
       }
   }

}


//영상처리를 활용한 지진 감지 시스템
int main(void)
{
   VideoCapture capture(0);

   if (!capture.isOpened())
   {
       cout << "카메라를 여는 데 실패했습니다." << endl;
       return -1;
   }

   Mat origin, past, now, result, edge, edgeRes, exEdge;

   uchar nB, nG, nR, pG, pB, pR;

   int dfrc = 30, nGray = 0, pGray = 0, stdNum = 0, exDegree = 3, count = 0 ,frame = 0;

   capture.read(edge);

   SetGray(edge);
   LRreverse(edge);

   Canny(edge, edgeRes, 50, 150, 3); //최초 프레임 엣지
   
   Mat element = getStructuringElement(MORPH_RECT, Size(2 * exDegree + 1, 2 * exDegree + 1));

   dilate(edgeRes, exEdge, element);

   stdNum = countNonZero(edgeRes);

   imshow("비교 사진", exEdge);
   waitKey(0);

   while (1) {
       capture.read(past);
       capture.read(now);
       capture.read(origin);

       LRreverse(origin);
       imshow("CCTV 영상", origin);

       uchar* past_data = past.data;
       uchar* now_data = now.data;

       int sameNum = 0;

       for (int row = 0; row < now.rows; row++)
       {
           for (int col = 0; col < now.cols; col++)
           {
               past_data[row * now.cols * 3 + col * 3] = now_data[row * past.cols * 3 + col * 3];
               past_data[row * now.cols * 3 + col * 3 + 1] = now_data[row * past.cols * 3 + col * 3 + 1];
               past_data[row * now.cols * 3 + col * 3 + 2] = now_data[row * past.cols * 3 + col * 3 + 2];

               pB = past_data[row * past.cols * 3 + col * 3];
               pG = past_data[row * past.cols * 3 + col * 3 + 1];
               pR = past_data[row * past.cols * 3 + col * 3 + 2];

               pGray = (uint(pB) + uint(pG) + uint(pR)) / 3;

               past_data[row * past.cols * 3 + col * 3] = uchar(pGray);
               past_data[row * past.cols * 3 + col * 3 + 1] = uchar(pGray);
               past_data[row * past.cols * 3 + col * 3 + 2] = uchar(pGray);
           }
       }

       capture.read(now);
       capture.read(result);

       uchar* now2_data = now.data;
       uchar* result_data = result.data;

       for (int row = 0; row < now.rows; row++)
       {
           for (int col = 0; col < now.cols; col++)
           {
               nB = now2_data[row * now.cols * 3 + col * 3];
               nG = now2_data[row * now.cols * 3 + col * 3 + 1];
               nR = now2_data[row * now.cols * 3 + col * 3 + 2];

               uchar pixel = exEdge.at<uchar>(row, col);

               nGray = (uint(nB) + uint(nG) + uint(nR)) / 3;

               pGray = past_data[row * past.cols * 3 + col * 3];

               now2_data[row * past.cols * 3 + col * 3] = uchar(nGray);
               now2_data[row * past.cols * 3 + col * 3 + 1] = uchar(nGray);
               now2_data[row * past.cols * 3 + col * 3 + 2] = uchar(nGray);

               if (nGray - pGray > dfrc || nGray - pGray < -dfrc) {
                   result_data[row * past.cols * 3 + col * 3] = 255;
                   result_data[row * past.cols * 3 + col * 3 + 1] = 255;
                   result_data[row * past.cols * 3 + col * 3 + 2] = 255;
               }
               else {
                   result_data[row * past.cols * 3 + col * 3] = 0;
                   result_data[row * past.cols * 3 + col * 3 + 1] = 0;
                   result_data[row * past.cols * 3 + col * 3 + 2] = 0;
               }

               if (pixel == 255 && result_data[row * past.cols * 3 + col * 3] == 255) {
                   sameNum++;
               }

           }
       }

       LRreverse(result);

       frame++;

       if (frame == 15) {
           frame = 0;
           count = 0;
       }

       double similarity = (double(sameNum) / double(stdNum)) * 100;

       if (similarity >= 100) {
           similarity = 99.9;
       }

       if (similarity > 55) {
           count++;
       }

       printf("건물이 흔들리고 있을 확률 : %.1lf %%\n", similarity);

       if (count > 4) {
           count = 0;
           for (int i = 0; i < 20; i++) {
               cout << "\033[31m지진이 발생했습니다!!!\033[0m" << endl;;
           }
       }

       imshow("영상을 활용한 지진 감지 시스템", result);

       if (waitKey(1) > 0) break;

   }

   destroyAllWindows();

   return 0;
}
