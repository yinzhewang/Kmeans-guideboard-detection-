#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>

using namespace cv;
using namespace std;

#define channels 3        //用于RGB通道的图像
#define k 13              //聚类的个数
#define _mindiff 1.0         //double型判断收敛距离
#define _Maxitr 20        //int最大迭代次数

class Color {
private:
	int vals[channels]{};
public:
	int operator[](const int i) const
	{
		return vals[i];
	}

	int& operator[](const int i)
	{
		return vals[i];
	}
	Color() = default;
	Color(int r, int g, int b) {
		vals[0] = r;
		vals[1] = g;
		vals[2] = b;
	}
};

typedef vector<vector<Color> > Image;
typedef vector<Color> Row;

// 函数声明
void MattoImage(Mat &src, Image &dst);
void RGB_kmeans(Image &src, Image &dst);
void ImagetoMat(Image &src, Mat &dst);
void Binary(Mat& src, Mat& dst);
void DrawGuideboard(Mat &inputImg, Mat foreImg);

int main(int argc, char** argv)
{
	Mat input_image = imread("D:\\KmeansTest\\test4.jpg");
	
	Mat output_image = Mat::zeros(input_image.size(), CV_8UC3);
	Mat dst;
	Image trans_in,trans_out;

	imshow("display", input_image);
	waitKey(0);

    MattoImage(input_image,  trans_in);
    RGB_kmeans(trans_in, trans_out);
	ImagetoMat(trans_out, output_image);
	Binary(output_image, dst);
	
	imshow("cluster picture", output_image);
	DrawGuideboard(input_image, dst);
	waitKey(0);

	return 0;
}

/*将导入的mat型变换到image*/
void MattoImage(Mat &src, Image &dst){
	for (int i = 0; i < src.rows; i++) {
		Row row;
		for (int j = 0; j < src.cols; j++) {
			auto temp = src.at<Vec3b>(i, j);
			row.emplace_back(Color(temp[2], temp[1], temp[0]));
		}
		dst.push_back(row);
	}
}

void RGB_kmeans(Image &src, Image &dst )
{
	int i, j, s, it,t,pos;
	double diff;
    double mindiff = _mindiff;
	int Maxitr = _Maxitr;

	assert(!src.empty());
	for (i = 1; i < src.size(); i++) {
		assert(src[i].size() == src[0].size());
	}

	int rows = src.size();
	int cols = src[0].size();

	Mat cluster(src.size(), src[0].size(), CV_8UC1);
	dst.clear();
	//给定初始质心
	double temp = src.size() / k;
	if (temp < 1||temp<0)
	{  
		printf("the input 'k' error.please try again!");
	}

	int interval = (src.size() - src.size() % k) / k;
	int krow[k] = {0};
	int kcol[k] = {0};//用于存放初始化随机k个质心点位置

	int kr[k];
	int kg[k];
	int kb[k];
	double fact_kr[k];
	double fact_kg[k];//用于存放k个质心点位置
	double fact_kb[k];
	int pre_kr[k];
	int pre_kg[k];//用于存放k个质心点位置坐标
	int pre_kb[k];

	for (i = 0; i < k; i++)
	{
		krow[i] =  i*interval; //选择了图片中心线上的k个点
		kcol[i]= (int)(src.size() / 2);
		kr[i] = src[krow[i]][kcol[i]][0];
		kg[i] = src[krow[i]][kcol[i]][1];
		kb[i] = src[krow[i]][kcol[i]][2];
	}
	//遍历图像中所有像素点，计算每个像素点和质心点的RGB距离，和哪个最近归到哪一类
	double distR, distG, distB,cur_dist,min_dist = 196000;
	double sumr, sumg ,sumb, num;
	for (it = 0; it < Maxitr; it++) {
		printf("iter has ready!! The iter time is %d  \n",(it+1));
		for (i = 0; i < rows; i++) {
			for (j = 0; j < cols; j++) {
				min_dist = 196000;
				for (s = 0; s < k; s++) {
				/*	printf("caculate distance!!\n");*/
					distR = pow(src[i][j][0] - kr[s], 2);
					distG = pow(src[i][j][1] - kg[s], 2);
					distB = pow(src[i][j][2] - kb[s], 2);
					cur_dist = distR + distG + distB;                         //计算颜色距离
					if (cur_dist < min_dist)
					{
						//说明对于当前点来说该点更靠近
						min_dist = cur_dist;
						cluster.at<uchar>(i,j) = s;    //对于这个像素点更靠近第s+1个聚类
					}
				}
			}
		}//完成聚类表 需要重新计算质心RGB!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!位置
		printf("完成聚类表 需要重新计算质心RGB位置!!\n");
		/*waitKey(0);*/
			for (s = 0; s < k; s++) {
				sumr = sumg = sumb = num = 0;
				for (i = 0; i < rows; i++) {
					for (j = 0; j < cols; j++) {
						if (cluster.at<uchar>(i, j) == s) {
							sumr += src[i][j][0];
							sumg += src[i][j][1];
							sumb += src[i][j][2];
							num++;
						}
					}
				}
				pre_kr[s] = kr[s];
				pre_kg[s] = kg[s];
				pre_kb[s] = kb[s];
				fact_kr[s] = sumr / num;
				fact_kg[s] = sumg / num;
				fact_kb[s] = sumb / num;
				kr[s] = int(fact_kr[s]);
				kg[s] = int(fact_kg[s]);
				kb[s] = int(fact_kb[s]);
			}
            //退出循环条件
			//waitKey(0);
			t = 0;
			for (s = 0; s < k; s++) {
				diff = pow((fact_kr[s] - pre_kr[s]), 2) + pow((fact_kg[s] - pre_kg[s]), 2)+pow((fact_kb[s] - pre_kb[s]),2);
				if (diff <= mindiff) {
					t++;                
				}
			}
			printf("cluster has renew!!\n");
			printf("t = %d , s = %d!!\n",t,s);
			for (i = 0; i < k; i++)
			{
				printf("( %d , %d , %d )\t ", kr[i], kg[i], kb[i]);
			}
			printf("\n");
			printf("\n");
			/*waitKey(0);*/
			if (t == k)
			{
				printf("kmeans has done,cluster complete!\n");
				break;
			}
		}
	printf("has gotto real dot, please wait a moment...\n");
	/*waitKey(0);*/
	//此时已经求得k个真正的聚类点，需要更新新的聚类表cluster[][]；
	for (i = 0; i < rows; i++) {
		for (j = 0; j < cols; j++) {
			min_dist = 196000;
			for (s = 0; s < k; s++) {
				distR = pow(src[i][j][0] - kr[s], 2);
				distG = pow(src[i][j][1] - kg[s], 2);
				distB = pow(src[i][j][2] - kb[s], 2);
				cur_dist = distR + distG + distB;
				if (cur_dist < min_dist) 
				{
					//说明对于当前点来说该点更靠近
					min_dist = cur_dist;
					cluster.at<uchar>(i, j) = s;    //对于这个像素点更靠近第s+1个聚类
				}
			}
		}
	}
	
	for (i = 0; i < rows; i++) {
		Row dst_row;
		for (j = 0; j < cols; j++) {
			//printf("1\n");
			//waitKey(0);
			
			//printf("2\n");
			//printf("r = %d , c = %d\n", r, c);
			//waitKey(0);
			Color cur_color = src[i][j];
            pos = cluster.at<uchar>(i, j);
			cur_color[0] = kr[pos];
			cur_color[1] = kg[pos];
			cur_color[2] = kb[pos];
			dst_row.push_back(cur_color);
			//printf("3\n");
			//waitKey(0);
		}
		dst.push_back(dst_row);
	}
	printf("即将转换图片!!!请按<任意键>开始产生聚类图片\n");
	waitKey(0);
}

void ImagetoMat(Image &src, Mat &dst) 
{
	for (unsigned long i = 0; i < src.size(); i++)
	{
		for (unsigned long j = 0; j < src[0].size(); j++)
		{
			dst.at<Vec3b>(i, j)[0] = src[i][j][2];
			dst.at<Vec3b>(i, j)[1] = src[i][j][1];
			dst.at<Vec3b>(i, j)[2] = src[i][j][0];
		}
	}  // convert Image to Mat
}

void Binary(Mat& src, Mat& dst) {
	int i, j, r, g, b,black = 0,white = 0;
	dst = Mat::zeros(src.size(), CV_8UC1);

	for (i = 0; i < src.rows; i++) {
		for (j = 0; j < src.cols; j++) {
			r = src.at<Vec3b>(i, j)[2];
			g = src.at<Vec3b>(i, j)[1];
			b = src.at<Vec3b>(i, j)[0];

			if( (b>g&&g>r&&((b-g)>30)&&r<100)|| (g>b&&b>r&&  r<100))   //((g - b)>30) &&
			{
				dst.at<uchar>(i, j) = 255;
				white++;
			}
			else
			{
				dst.at<uchar>(i, j) = 0;
				black++;
			}
		}
	}
	imshow("binary picture", dst);
}

void DrawGuideboard(Mat &inputImg, Mat foreImg)
{
	vector<vector<Point>> contours_set;//保存轮廓提取后的点集及拓扑关系  

	findContours(foreImg, contours_set, RETR_EXTERNAL, CHAIN_APPROX_NONE);

	Mat result0;
	Scalar holeColor;
	Scalar externalColor;

	vector<vector<Point>>::iterator iter = contours_set.begin();
	for (; iter != contours_set.end(); )   //迭代器循环
	{
		Rect rect = boundingRect(*iter);

		/*	float radius;
		Point2f center;
		minEnclosingCircle(*iter, center, radius);*/

		if(( rect.area()> 0)&&(rect.width> 30) && (rect.height> 30))
		{
			rectangle(inputImg, rect, Scalar(0, 255, 0),5);  //scalar表示是什么颜色去框选
			++iter;
		}
		else
		{
			iter = contours_set.erase(iter);
		}
	}
	imshow("Detect Guideboard", inputImg);
}