// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <queue>
#include <random>
#include <vector>
#include <stack>
#include <stdio.h>

using namespace std;

default_random_engine gen;
uniform_int_distribution<int> d(0, 255);

void testOpenImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image", src);
		waitKey();
	}
}

void grayImage() {
	Mat img1 = Mat(50, 70, CV_8UC1);
	for (int i = 0; i < img1.rows; i++) for (int j = 0; j < img1.cols; j++) img1.at<uchar>(i, j) = 128;
	imshow("gray-image", img1);
	waitKey();
}


void imagine()
{
	Vec3b culori[] = { Vec3b(0,0,255),Vec3b(80,127,250), Vec3b(0,255,255), Vec3b(0,0,255), Vec3b(0,255,0), Vec3b(255,0,0), Vec3b(130,0,70),Vec3b(230,130,238) };
	Mat img(256, 256, CV_8UC3);
	for (int i = 0; i < img.rows; i++) for (int j = 0; j < img.cols; j++) {
		Vec3b pixel;
		pixel = culori[i % 10];
		img.at< Vec3b>(i, j) = pixel;

	}
	imshow("imageg", img);
	waitKey();


}

void aduna() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		imshow("src", src);

		int val = 70;
		Mat dst = Mat(src.rows, src.cols, CV_8UC1);
		for (int i = 0; i < src.rows; i++) for (int j = 0; j < src.cols; j++) if (src.at<uchar>(i, j) + val > 255) dst.at<uchar>(i, j) = 255;
		else if (src.at<uchar>(i, j) + val < 0) dst.at<uchar>(i, j) = 0;
		else dst.at<uchar>(i, j) = src.at<uchar>(i, j) + val;
		imshow("dst", dst);
		waitKey();
	}
}

void color2gray1() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_COLOR);
		imshow("src", src);
		Mat dst = Mat(src.rows, src.cols, CV_8UC1);
		Mat dst1 = Mat(src.rows, src.cols, CV_8UC1);
		for (int i = 0; i < src.rows; i++) for (int j = 0; j < src.cols; j++) {
			Vec3b pixel = src.at<Vec3b>(i, j);
			dst.at<uchar>(i, j) = (pixel[2] + pixel[1] + pixel[0]) / 3;
			dst1.at<uchar>(i, j) = 0.21*pixel[2] + 0.71*pixel[1] + 0.072*pixel[0];
		}
		imshow("dst", dst);
		imshow("dst1", dst1);
		waitKey();
	}
}


void binarizare() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		imshow("src", src);
		Mat dst = Mat(src.rows, src.cols, CV_8UC1);
		for (int i = 0; i < src.rows; i++) for (int j = 0; j < src.cols; j++) {
			if (src.at<uchar>(i, j) < 70) dst.at<uchar>(i, j) = 0;
			else dst.at<uchar>(i, j) = 255;
		}
		imshow("dst", dst);
		waitKey();
	}
}


void split_rgb() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_COLOR);
		imshow("src", src);
		Mat dst = Mat(src.rows, src.cols, CV_8UC1);
		Mat dst1 = Mat(src.rows, src.cols, CV_8UC1);
		Mat dst2 = Mat(src.rows, src.cols, CV_8UC1);
		for (int i = 0; i < src.rows; i++) for (int j = 0; j < src.cols; j++) {
			Vec3b pixel = src.at<Vec3b>(i, j);
			dst.at<uchar>(i, j) = pixel[2];
			dst1.at<uchar>(i, j) = pixel[1];
			dst2.at<uchar>(i, j) = pixel[0];
		}
		imshow("rosu", dst);
		imshow("verde", dst1);
		imshow("albastru", dst2);

		waitKey();
	}
}

void split_rgb_color() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_COLOR);
		imshow("src", src);
		Mat dst = Mat(src.rows, src.cols, CV_8UC3);
		Mat dst1 = Mat(src.rows, src.cols, CV_8UC3);
		Mat dst2 = Mat(src.rows, src.cols, CV_8UC3);
		for (int i = 0; i < src.rows; i++) for (int j = 0; j < src.cols; j++) {
			Vec3b pixel = src.at<Vec3b>(i, j);
			Vec3b pixel1, pixel2, pixel3;

			pixel1[2] = pixel[2];
			pixel1[1] = pixel1[0] = 0;

			pixel2[1] = pixel[1];
			pixel2[2] = pixel2[0] = 0;

			pixel3[0] = pixel[0];
			pixel3[2] = pixel3[1] = 0;
			dst.at<Vec3b>(i, j) = pixel1;
			dst1.at<Vec3b>(i, j) = pixel2;
			dst2.at<Vec3b>(i, j) = pixel3;
		}

		imshow("rosu", dst);
		imshow("verde", dst1);
		imshow("albastru", dst2);

		waitKey();
	}
}

void split_rgb_to_hsv() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_COLOR);
		imshow("src", src);
		Mat dst = Mat(src.rows, src.cols, CV_8UC1);
		Mat dst1 = Mat(src.rows, src.cols, CV_8UC1);
		Mat dst2 = Mat(src.rows, src.cols, CV_8UC1);
		Mat dst3 = Mat(src.rows, src.cols, CV_8UC3);
		for (int i = 0; i < src.rows; i++) for (int j = 0; j < src.cols; j++) {
			Vec3b pixel = src.at<Vec3b>(i, j);
			float r, g, b;
			r = pixel[2] / 255.0;
			g = pixel[1] / 255.0;
			b = pixel[0] / 255.0;
			float M, m;
			M = max(max(r, g), b);
			m = min(min(r, g), b);
			float V, S, H, C;
			C = M - m;
			V = M;
			if (V != 0) S = C / V;
			else
				S = 0;
			if (C != 0) {
				if (M == r) H = 60 * (g - b) / C;
				if (M == g) H = 120 + 60 * (b - r) / C;
				if (M == b) H = 240 + 60 * (r - g) / C;
			}
			else H = 0;
			if (H < 0) H = H + 360;
			float H_norm, S_norm, V_norm;
			H_norm = H * 255 / 360;
			S_norm = S * 255;
			V_norm = V * 255;
			dst.at<uchar>(i, j) = H_norm;
			dst1.at<uchar>(i, j) = S_norm;
			dst2.at<uchar>(i, j) = V_norm;

			Vec3b pixelfinal;
			pixelfinal[2] = H_norm;
			pixelfinal[1] = S_norm;
			pixelfinal[0] = V_norm;
			dst3.at<Vec3b>(i, j) = pixelfinal;

		}

		imshow("H", dst);
		imshow("S", dst1);
		imshow("V", dst2);
		imshow("HSV", dst3);

		waitKey();
	}
}

Mat split_rgb_to_hsv(Mat src) {

	Mat dst = Mat(src.rows, src.cols, CV_8UC1);
	Mat dst1 = Mat(src.rows, src.cols, CV_8UC1);
	Mat dst2 = Mat(src.rows, src.cols, CV_8UC1);
	Mat dst3 = Mat(src.rows, src.cols, CV_8UC3);
	for (int i = 0; i < src.rows; i++) for (int j = 0; j < src.cols; j++) {
		Vec3b pixel = src.at<Vec3b>(i, j);
		float r, g, b;
		r = pixel[2] / 255.0;
		g = pixel[1] / 255.0;
		b = pixel[0] / 255.0;
		float M, m;
		M = max(max(r, g), b);
		m = min(min(r, g), b);
		float V, S, H, C;
		C = M - m;
		V = M;
		if (V != 0) S = C / V;
		else
			S = 0;
		if (C != 0) {
			if (M == r) H = 60 * (g - b) / C;
			if (M == g) H = 120 + 60 * (b - r) / C;
			if (M == b) H = 240 + 60 * (r - g) / C;
		}
		else H = 0;
		if (H < 0) H = H + 360;
		float H_norm, S_norm, V_norm;
		H_norm = H * 255 / 360;
		S_norm = S * 255;
		V_norm = V * 255;
		dst.at<uchar>(i, j) = H_norm;
		dst1.at<uchar>(i, j) = S_norm;
		dst2.at<uchar>(i, j) = V_norm;
		Vec3b pixelfinal;
		pixelfinal[2] = H_norm;
		pixelfinal[1] = S_norm;
		pixelfinal[0] = V_norm;
		dst3.at<Vec3b>(i, j) = pixelfinal;

	}
	return dst3;
}

int isInside(Mat img, int i, int j) {

	if (i >= 0 && i < img.rows) {
		if (j >= 0 && j < img.cols) {
			return 1;
		}
	}
	return 0;

}

void verificare() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat img = imread(fname, CV_LOAD_IMAGE_COLOR);
		if (isInside(img, 100, 100) == 1) std::cout << "Adevarat";
		else std::cout << "Fals";
	}
}



void inmultire() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		imshow("src", src);

		int val = 2;
		Mat dst = Mat(src.rows, src.cols, CV_8UC1);
		for (int i = 0; i < src.rows; i++) for (int j = 0; j < src.cols; j++) if (src.at<uchar>(i, j) * val > 255) dst.at<uchar>(i, j) = 255;
		else  if (src.at<uchar>(i, j) * val < 0) dst.at<uchar>(i, j) = 0;
		else dst.at<uchar>(i, j) = src.at<uchar>(i, j) * val;
		imshow("dst", dst);
		imwrite("noua imagine", dst);
		waitKey();
	}
}


void adunarosu() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_COLOR);
		imshow("src", src);

		int val = 10;
		Mat dst = Mat(src.rows, src.cols, CV_8UC3);
		for (int i = 0; i < src.rows; i++) for (int j = 0; j < src.cols; j++) if (src.at<Vec3b>(i, j)[2] + val > 255) dst.at<Vec3b>(i, j)[2] = 255;
		else if (src.at<Vec3b>(i, j)[2] + val < 0) dst.at<Vec3b>(i, j)[2] = 0;
		else dst.at<Vec3b>(i, j)[2] = src.at<Vec3b>(i, j)[2] + val;
		imshow("dst", dst);
	}
}


void inversa() {
	float vals[] = { 1,2,3,4,5,6,7,8.0,1 };
	Mat m = Mat(3, 3, CV_32FC1, vals);
	//std::cout << m << std::endl;
	Mat mi = m.inv();
	std::cout << mi;
	waitKey();
}

void scade() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		imshow("src", src);

		int val = 70;
		Mat dst = Mat(src.rows, src.cols, CV_8UC1);
		for (int i = 0; i < src.rows; i++) for (int j = 0; j < src.cols; j++) if (src.at<uchar>(i, j) + val < 0) dst.at<uchar>(i, j) = 0;
		else if (src.at<uchar>(i, j) + val > 255) dst.at<uchar>(i, j) = 255;
		else dst.at<uchar>(i, j) = src.at<uchar>(i, j) - val;
		imshow("dst", dst);
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName) == 0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName, "bmp");
	while (fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(), src);
		if (waitKey() == 27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", CV_LOAD_IMAGE_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, CV_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}

void negative_image() {
	Mat img = imread("Images/cameraman.bmp",
		CV_LOAD_IMAGE_GRAYSCALE);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			img.at<uchar>(i, j) = 255 - img.at<uchar>(i, j);
		}
	}
	imshow("negative image", img);
	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				uchar neg = 255 - val;
				dst.at<uchar>(i, j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// the fastest approach using the “diblook style”
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		int w = (int)src.step; // no dword alignment is done !!!
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height, width, CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i, j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i, j) = (r + g + b) / 3;
			}
		}

		imshow("input image", src);
		imshow("gray image", dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// Componentele d eculoare ale modelului HSV
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, CV_BGR2HSV);

		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				int hi = i * width * 3 + j * 3;
				int gi = i * width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;		// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}


void testResize()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1, dst2;
		//without interpolation
		resizeImg(src, dst1, 320, false);
		//with interpolation
		resizeImg(src, dst2, 320, true);
		imshow("input image", src);
		imshow("resized image (without interpolation)", dst1);
		imshow("resized image (with interpolation)", dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src, dst, gauss;
		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int)k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss, dst, pL, pH, 3);
		imshow("input image", src);
		imshow("canny", dst);
		waitKey();
	}
}

void testVideoSequence()
{
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}

	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, CV_BGR2GRAY);
		Canny(grayFrame, edges, 40, 100, 3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = cvWaitKey(0);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n");
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];

	// video resolution
	Size capS = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;

		imshow(WIN_SRC, frame);

		c = cvWaitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115) { //'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess)
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}



/* Histogram display function - display a histogram using bars (simlilar to L3 / PI)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/
void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i < hist_cols; i++)
		if (hist[i] > max_hist)
			max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}

void histograma() {
	int hist[256] = { 0 };
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);

		for (int i = 0; i < img.rows; i++) for (int j = 0; j < img.cols; j++) {
			hist[img.at<uchar>(i, j)]++;
		}
		showHistogram("Histograma", hist, img.cols, img.rows);
		waitKey();
	}
}


void fdp() {
	int hist[256] = { 0 };
	float fdp[256] = { 0.0 };
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);

		for (int i = 0; i < img.rows; i++) for (int j = 0; j < img.cols; j++) {
			hist[img.at<uchar>(i, j)]++;
		}

		for (int i = 0; i < 256; i++) {
			fdp[i] = (float)hist[i] / (img.rows*img.cols);
		}

		printf("FDP:\n");

		for (int i = 0; i < 256; i++) printf("%f ", fdp[i]);



		showHistogram("Histograma", hist, img.cols, img.rows);
		waitKey();
	}
}

void histacumulatoare(int n) {

	int *hist;
	hist = (int *)malloc(sizeof(int)*n);
	for (int i = 0; i < n; i++) hist[i] = 0;
	int parti = 255 / n;
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);

		for (int i = 0; i < img.rows; i++) for (int j = 0; j < img.cols; j++) {
			if (img.at<uchar>(i, j) / parti == n) hist[n - 1]++;
			else
				hist[img.at<uchar>(i, j) / parti]++;
		}

		printf("Histograma cu mai putine acumulatoare:\n");
		for (int i = 0; i < n; i++) printf("%d ", hist[i]);
		waitKey();
	}
}

void histograma_3_canale() {
	char fname[MAX_PATH];
	int hist[256] = { 0 }, hist1[256] = { 0 }, hist2[256] = { 0 };
	while (openFileDlg(fname))
	{
		Mat img = imread(fname, CV_LOAD_IMAGE_COLOR);

		for (int i = 0; i < img.rows; i++) for (int j = 0; j < img.cols; j++) {
			Vec3b pixel = img.at<Vec3b>(i, j);
			hist2[pixel[2]]++;
			hist1[pixel[1]]++;
			hist[pixel[0]]++;

		}
		showHistogram("Histograma blue", hist, img.cols, img.rows);
		showHistogram("Histograma green", hist1, img.cols, img.rows);
		showHistogram("Histograma red", hist2, img.cols, img.rows);
		waitKey();
	}
}

void determinare_praguri_multiple() {
	int hist[256] = { 0 };
	float fdp[256] = { 0.0 };
	float TH = 0.0003;
	int WH = 5;
	char fname[MAX_PATH];
	float v;
	int praguri[256];
	int contor = 1;
	praguri[0] = 0;
	while (openFileDlg(fname))
	{
		Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat dst = Mat(img.rows, img.cols, CV_8UC1);

		for (int i = 0; i < img.rows; i++) for (int j = 0; j < img.cols; j++) {
			hist[img.at<uchar>(i, j)]++;
		}

		for (int i = 0; i < 256; i++) {
			fdp[i] = (float)hist[i] / (img.rows*img.cols);
		}
		int suma = 0;
		for (int k = WH; k < 256 - WH; k++) {
			int ok = 0;
			for (int i = k - WH; i <= k + WH; i++) {
				suma = suma + fdp[i];
				if (fdp[k] < fdp[i]) ok = 1;
			}
			v = (float)suma / (2 * WH + 1);
			if (fdp[k] > v + TH && ok == 0) {
				praguri[contor] = k; contor++;
			}
		}

		praguri[contor] = 255;
		contor++;

		for (int i = 0; i < contor; i++) printf("%d ", praguri[i]);

		for (int i = 0; i < img.rows; i++) for (int j = 0; j < img.cols; j++) {
			for (int k = 0; k < contor - 1; k++) if (img.at<uchar>(i, j) >= praguri[k] && img.at<uchar>(i, j) < praguri[k + 1]) {
				if (img.at<uchar>(i, j) < (praguri[k] + praguri[k + 1]) / 2) dst.at<uchar>(i, j) = praguri[k];
				else dst.at<uchar>(i, j) = praguri[k + 1];
			}
		}


		imshow("src", img);
		imshow("dst", dst);
		waitKey();
	}
}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{

	Mat* src = (Mat*)param;
	if (event == CV_EVENT_LBUTTONDOWN)
	{
		printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
			x, y,
			(int)(*src).at<Vec3b>(y, x)[2],
			(int)(*src).at<Vec3b>(y, x)[1],
			(int)(*src).at<Vec3b>(y, x)[0]);
	}
}

void MyCallBackFunc1(int event, int x, int y, int flags, void* param)
{

	Mat* img = (Mat*)param;
	if (event == CV_EVENT_LBUTTONDOWN)
	{
		Vec3b culoare = img->at<Vec3b>(y, x);
		Mat dst = Mat(img->rows, img->cols, CV_8UC3);
		Mat proj = Mat(img->rows, img->cols, CV_8UC3);
		float aria = 0, sumri = 0, sumci = 0, sum = 0, sum1 = 0, sum2 = 0, T = 0;
		float numitor = 0, numarator = 0, numa = 0, numi = 0, alungire = 0, perimetru = 0;
		float cmax = INT_MIN, cmin = INT_MAX, rmax = INT_MIN, rmin = INT_MAX, R = 0;
		float ri = 0, ci = 0;


		for (int i = 0; i < img->rows; i++) {
			for (int j = 0; j < img->cols; j++) {

				if (img->at<Vec3b>(i, j) == culoare) {

					aria++;

					sumri += i;
					sumci += j;

				}
			}
		}

		ri = sumri / aria;
		ci = sumci / aria;

		for (int i = 0; i < img->rows; i++) {
			for (int j = 0; j < img->cols; j++) {

				if (img->at<Vec3b>(i, j) == culoare) {

					sum += (i - ri) * (j - ci);
					sum1 += (j - ci) * (j - ci);
					sum2 += (i - ri) * (i - ci);


					if (i < rmin)
						rmin = i;
					if (i > rmax)
						rmax = i;
					if (j < cmin)
						cmin = j;
					if (j > cmax)
						cmax = j;

				}


				if (img->at<Vec3b>(i, j) == culoare && (img->at<Vec3b>(i + 1, j) != culoare || img->at<Vec3b>(i - 1, j) != culoare || img->at<Vec3b>(i, j + 1) != culoare || img->at<Vec3b>(i, j - 1) != culoare || img->at<Vec3b>(i + 1, j + 1) != culoare || img->at<Vec3b>(i + 1, j - 1) != culoare || img->at<Vec3b>(i - 1, j + 1) != culoare || img->at<Vec3b>(i - 1, j - 1) != culoare)) {

					perimetru++;
					dst.at<Vec3b>(i, j) = culoare;

				}

				dst.at<Vec3b>(ri, ci) = 0;
				dst.at<Vec3b>(ri, ci + 1) = 0;
				dst.at<Vec3b>(ri, ci - 1) = 0;
				dst.at<Vec3b>(ri + 1, ci) = 0;
				dst.at<Vec3b>(ri - 1, ci) = 0;

			}
		}

		numarator = 2 * sum;
		numitor = sum1 - sum2;
		alungire = atan2(numarator, numitor) / 2;

		T = 4 * PI * (aria / (perimetru * perimetru));
		R = (cmax - cmin + 1) / (rmax - rmin + 1);

		line(dst, Point(ci + 75 * cos(alungire), ri + 75 * sin(alungire)), Point(ci - 75 * cos(alungire), ri - 75 * sin(alungire)), Scalar(0, 0, 0));


		for (int i = 0; i < img->rows; i++) {

			int nr = 0;
			for (int j = 0; j < img->cols; j++) {

				if (img->at<Vec3b>(i, j) == culoare)
					++nr;
			}

			for (int j = 0; j < nr; j++) {
				proj.at<Vec3b>(i, j) = culoare;
			}
		}



		for (int j = 0; j < img->cols; j++) {

			int nr = 0;
			for (int i = 0; i < img->rows; i++) {

				if (img->at<Vec3b>(i, j) == culoare)
					++nr;
			}

			for (int i = img->rows - nr + 1; i < img->rows; i++) {
				proj.at<Vec3b>(i, j) = culoare;
			}
		}




		printf("Aria: %f\n", aria);
		printf("Centru de masa: (%f,%f)\n", ri, ci);
		printf("Alungirea: %f\n", alungire);
		printf("Perimetru: %f\n", perimetru);
		printf("Factor de subtiere: %f\n", T);
		printf("Elongatia: %f\n", R);

		imshow("dst", dst);
		imshow("proj", proj);
	}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

void lab4()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc1, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}



void ariemaimica(float th_arie, float phi_low, float phi_high) {
	char fname[MAX_PATH];
	Vec3b culori[10] = { {255,0,0},{0,255,0 },{0,0,255},{128,128,128},{192,128,0},{255,0,255},{0,64,128} };
	while (openFileDlg(fname))
	{
		Mat img = imread(fname, CV_LOAD_IMAGE_COLOR);
		Mat dst = Mat(img.rows, img.cols, CV_8UC3);

		for (int k = 0; k < 7; k++) {
			float aria = 0;
			float sumri = 0, sumci = 0, ri = 0, ci = 0, sum = 0, sum1 = 0, sum2 = 0, numarator = 0, numitor = 0, alungire = 0;
			for (int i = 0; i < img.rows; i++)
				for (int j = 0; j < img.cols; j++) {
					if (img.at<Vec3b>(i, j) == culori[k]) {
						aria++;
						sumri = sumri + i;
						sumci = sumci + j;
					}
				}

			ri = sumri / aria;
			ci = sumci / aria;

			for (int i = 0; i < img.rows; i++)
				for (int j = 0; j < img.cols; j++) {
					if (img.at<Vec3b>(i, j) == culori[k]) {
						sum += (i - ri) * (j - ci);
						sum1 += (j - ci) * (j - ci);
						sum2 += (i - ri) * (i - ci);
					}
				}

			numarator = 2 * sum;
			numitor = sum1 - sum2;
			alungire = atan2(numarator, numitor) / 2;

			printf("%f\n", aria);
			printf("%f\n", alungire);

			if (aria < th_arie && alungire>phi_low && alungire < phi_high) {
				for (int i = 0; i < img.rows; i++)
					for (int j = 0; j < img.cols; j++)
						if (img.at<Vec3b>(i, j) == culori[k]) dst.at<Vec3b>(i, j) = img.at<Vec3b>(i, j);
			}
		}



		imshow("src", img);
		imshow("dst", dst);
		waitKey();
	}


}


void generare_culori(Mat img, Mat labels) {

	Mat dst = Mat(img.rows, img.cols, CV_8UC3);

	int maxLabel = INT_MIN;

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {

			if (labels.at<int>(i, j) > maxLabel)
				maxLabel = labels.at<int>(i, j);
		}
	}

	vector<Vec3b> culori(maxLabel + 1);
	for (int k = 0; k < maxLabel; k++) {

		uchar r = d(gen);
		uchar g = d(gen);
		uchar b = d(gen);
		culori.at(k) = Vec3b(b, g, r);
	}


	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {

			if (labels.at<int>(i, j) > 0) {

				dst.at<Vec3b>(i, j) = culori.at(labels.at<int>(i, j));

			}

			else
				dst.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
		}
	}

	imshow("dst", dst);
	waitKey();
}



void BFS() {

	char fname[MAX_PATH];

	while (openFileDlg(fname))
	{
		Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int label = 0;
		Mat labels = Mat(img.rows, img.cols, CV_32SC1);

		for (int i = 0; i < labels.rows; i++) for (int j = 0; j < labels.cols; j++) labels.at<int>(i, j) = 0;

		vector<int> di, dj;
		int mode;
		printf("Alegere vecinatate N4 sau N8 (0 sau 1)?");
		scanf("%d", &mode);

		if (mode == 1) {
			di = { 0, -1, -1, -1,  0 , 1, 1, 1 };
			dj = { 1,  1,  0, -1, -1, -1, 0, 1 };
		}
		else if (mode == 0) {
			di = { 0, -1, 0 , 1 };
			dj = { 1,  0, -1, 0 };
		}

		for (int i = 0; i < img.rows; i++) {
			for (int j = 0; j < img.cols; j++) {

				if (img.at<uchar>(i, j) == 0 && labels.at<int>(i, j) == 0) {

					label++;
					queue<Point2i> Q;
					labels.at<int>(i, j) = label;
					Q.push({ i,j });

					while (!Q.empty()) {

						Point2i q = Q.front(); Q.pop();

						for (int k = 0; k < dj.size(); k++) {

							Point2i neighbor(q.x + di[k], q.y + dj[k]);

							if (isInside(img, neighbor.x, neighbor.y) && img.at<uchar>(neighbor.x, neighbor.y) == 0 && labels.at<int>(neighbor.x, neighbor.y) == 0) {

								labels.at<int>(neighbor.x, neighbor.y) = label;
								Q.push(neighbor);
							}
						}
					}



				}
			}

		}
		generare_culori(img, labels);
	}
}


void DFS() {

	char fname[MAX_PATH];

	while (openFileDlg(fname))
	{
		Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int label = 0;
		Mat labels = Mat(img.rows, img.cols, CV_32SC1);

		for (int i = 0; i < labels.rows; i++) for (int j = 0; j < labels.cols; j++) labels.at<int>(i, j) = 0;

		vector<int> di, dj;
		int mode;
		printf("Alegere vecinatate N4 sau N8 (0 sau 1)?");
		scanf("%d", &mode);

		if (mode == 1) {
			di = { 0, -1, -1, -1,  0 , 1, 1, 1 };
			dj = { 1,  1,  0, -1, -1, -1, 0, 1 };
		}
		else if (mode == 0) {
			di = { 0, -1, 0 , 1 };
			dj = { 1,  0, -1, 0 };
		}

		for (int i = 0; i < img.rows; i++) {
			for (int j = 0; j < img.cols; j++) {

				if (img.at<uchar>(i, j) == 0 && labels.at<int>(i, j) == 0) {

					label++;
					stack<Point2i> Q;
					labels.at<int>(i, j) = label;
					Q.push({ i,j });

					while (!Q.empty()) {

						Point2i q = Q.top(); Q.pop();

						for (int k = 0; k < dj.size(); k++) {

							Point2i neighbor(q.x + di[k], q.y + dj[k]);

							if (isInside(img, neighbor.x, neighbor.y) && img.at<uchar>(neighbor.x, neighbor.y) == 0 && labels.at<int>(neighbor.x, neighbor.y) == 0) {

								labels.at<int>(neighbor.x, neighbor.y) = label;
								Q.push(neighbor);
							}
						}
					}


				}
			}

		}
		generare_culori(img, labels);
	}
}


void parcurgere_dubla() {

	char fname[MAX_PATH];

	while (openFileDlg(fname))
	{
		Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int label = 0, x = 0;
		Mat labels = Mat(img.rows, img.cols, CV_32SC1);
		labels.setTo(0);

		vector<vector<int>> edges;

		vector<int> di, dj;

		int mode;
		printf("Alege vecinatatea N4 sau N8 (0 sau 1) ?");
		scanf("%d", &mode);

		if (mode == 1) {
			di = { 0, -1, -1, -1,  0 , 1, 1, 1 };
			dj = { 1,  1,  0, -1, -1, -1, 0, 1 };
		}
		else if (mode == 0) {
			di = { 0, -1, 0 , 1 };
			dj = { 1,  0, -1, 0 };
		}

		for (int i = 0; i < img.rows; i++) {
			for (int j = 0; j < img.cols; j++) {

				if (img.at<uchar>(i, j) == 0 && labels.at<int>(i, j) == 0) {
					vector<int> L;

					for (int k = 0; k < dj.size(); k++) {

						Point2i neighbor = { i + di[k], j + dj[k] };
						if (isInside(img, neighbor.x, neighbor.y))
							if (labels.at<int>(neighbor.x, neighbor.y) > 0)
								L.push_back(labels.at<int>(neighbor.x, neighbor.y));
					}

					if (L.size() == 0) {
						label++;
						labels.at<int>(i, j) = label;
						edges.resize(label + 1, vector<int>(label + 1));
					}
					else
					{
						int x = INT_MAX;
						for (int nr = 0; nr < L.size(); nr++) {
							if (x > L[nr]) x = L[nr];
						}

						labels.at<int>(i, j) = x;

						for (int y = 0; y < L.size(); y++) {
							if (L[y] != x) {
								edges[x].push_back(L[y]);
								edges[L[y]].push_back(x);
							}
						}
					}
				}
			}
		}


		int newLabel = 0;
		vector<int> newLabels(label + 1);
		for (int i = 0; i < newLabels.size(); i++)
		{
			newLabels[i] = 0;
		}

		for (int i = 1; i < label + 1; i++) {

			if (newLabels[i] == 0) {
				newLabel++;
				queue<int> Q;
				newLabels[i] = newLabel;
				Q.push(i);

				while (!Q.empty()) {

					x = Q.front();
					Q.pop();

					for (int y = 0; y < edges[x].size(); y++) {
						if (edges[x][y] != 0) {
							if (newLabels[edges[x][y]] == 0) {
								newLabels[edges[x][y]] = newLabel;
								Q.push(edges[x][y]);
							}
						}
					}
				}
			}
		}



		for (int i = 0; i < img.rows; i++) {
			for (int j = 0; j < img.cols; j++) {

				labels.at<int>(i, j) = newLabels[labels.at<int>(i, j)];
			}
		}

		generare_culori(img, labels);
	}

}


void contur()
{

	char fname[MAX_PATH];

	FILE* f = fopen("construct.txt", "w+");

	while (openFileDlg(fname))
	{
		Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat dst = Mat(img.rows, img.cols, CV_8UC1);

		for (int i = 0; i < dst.rows; i++)
			for (int j = 0; j < dst.cols; j++) dst.at<uchar>(i, j) = 255;

		int di[8] = { 0, -1, -1, -1,  0 , 1, 1, 1 };
		int dj[8] = { 1,  1,  0, -1, -1, -1, 0, 1 };

		vector<int> dirs;
		vector<Point2i> pixels;

		bool flag = false;
		for (int i = 0; i < img.rows && !flag; i++) {
			for (int j = 0; j < img.cols && !flag; j++) {

				if (img.at<uchar>(i, j) == 0) {

					int dir = 7;
					dirs.push_back(dir);
					pixels.push_back({ i,j });
					dst.at<uchar>(i, j) = 0;
					flag = true;
				}
			}
		}

		flag = false;
		int i = 0;
		while (!flag) {

			int n = pixels.size();

			if (n > 2 && pixels.at(0) == pixels.at(n - 2) && pixels.at(1) == pixels.at(n - 1))
			{
				flag = true;
			}


			Point2i pixel = pixels.at(i);
			int dir = dirs.at(i);
			bool flag2 = false;
			for (int k = 0; k < 8 && !flag2; k++) {

				int dirNou;
				if (dir % 2 == 0)
					dirNou = (dir + 7) % 8;
				else
					dirNou = (dir + 6) % 8;


				Point2i neighbor(pixel.x + di[(dirNou + k) % 8], pixel.y + dj[(dirNou + k) % 8]);

				if (isInside(img, neighbor.x, neighbor.y) && img.at<uchar>(neighbor.x, neighbor.y) == 0) {


					dirs.push_back((dirNou + k) % 8);
					pixels.push_back(neighbor);
					i++;
					dst.at<uchar>(neighbor.x, neighbor.y) = 0;
					flag2 = true;
				}
			}

		}

		printf("Codul inlantuit: ");

		for (int i = 0; i < dirs.size(); i++) {

			printf("%d ", dirs[i]);
			fprintf(f, "%d%s", dirs[i], " ");


		}

		printf("\nDerivata: ");
		fprintf(f, "%s", "\n");

		for (int i = 0; i < dirs.size(); i++) {

			if (i > 0) {
				int derivata = (dirs.at(i) - dirs.at(i - 1) + 8) % 8;
				printf("%d ", derivata);
				fprintf(f, "%d%s", derivata, " ");
			}
		}


		fclose(f);

		imshow("dst", dst);
		waitKey();
	}
}


void reconstructie() {

	FILE* f = fopen("reconstruct.txt", "r");

	int di[8] = { 0, -1, -1, -1,  0 , 1, 1, 1 };
	int dj[8] = { 1,  1,  0, -1, -1, -1, 0, 1 };

	Mat dst = Mat(500, 750, CV_8UC1);

	dst.setTo(255);


	if (f != NULL) {

		int x, y, n;
		fscanf(f, "%d", &x);
		fscanf(f, "%d", &y);
		fscanf(f, "%d", &n);

		dst.at<uchar>(x, y) = 0;

		for (int i = 0; i < n; i++) {
			int dir;
			fscanf(f, "%d", &dir);
			Point2i pixel(x + di[dir], y + dj[dir]);
			dst.at<uchar>(pixel.x, pixel.y) = 0;
			x = pixel.x;
			y = pixel.y;
		}
	}

	fclose(f);

	imshow("imagine reconstruita", dst);
	waitKey();
}

void dilatare(int n) {

	char fname[MAX_PATH];

	int di[8] = { 0, -1, -1, -1,  0 , 1, 1, 1 };
	int dj[8] = { 1,  1,  0, -1, -1, -1, 0, 1 };

	while (openFileDlg(fname))
	{

		Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat imagine = img.clone();
		Mat dst = Mat(img.rows, img.cols, CV_8UC1);

		while (n > 0) {

			for (int i = 0; i < img.rows; i++)
				for (int j = 0; j < img.cols; j++)
				{
					if (img.at<uchar>(i, j) == 0) {

						dst.at<uchar>(i, j) = 0;

						for (int k = 0; k < 8; k++) {
							Point2i neighbor(i + di[k], j + dj[k]);

							if (isInside(img, neighbor.x, neighbor.y))
								dst.at<uchar>(neighbor.x, neighbor.y) = 0;
						}
					}

					else {
						dst.at<uchar>(i, j) = 255;
					}
				}

			n--;
			img = dst.clone();

		}
		imshow("img", imagine);
		imshow("dilatare", dst);
		waitKey();
	}
}

void eroziune(int n) {

	char fname[MAX_PATH];
	int di[8] = { 0, -1, -1, -1,  0 , 1, 1, 1 };
	int dj[8] = { 1,  1,  0, -1, -1, -1, 0, 1 };

	while (openFileDlg(fname))
	{

		Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat imagine = img.clone();
		Mat dst = Mat(img.rows, img.cols, CV_8UC1);

		while (n > 0) {

			for (int i = 0; i < img.rows; i++)
				for (int j = 0; j < img.cols; j++)
				{
					if (img.at<uchar>(i, j) == 0) {

						bool flag = false;
						for (int k = 0; k < 8 && !flag; k++) {
							Point2i neighbor(i + di[k], j + dj[k]);
							if (!isInside(img, neighbor.x, neighbor.y) || img.at<uchar>(neighbor.x, neighbor.y) != 0) flag = true;
						}


						if (!flag)	dst.at<uchar>(i, j) = 0;

						else {
							dst.at<uchar>(i, j) = 255;
						}

					}

					else {
						dst.at<uchar>(i, j) = 255;
					}
				}

			n--;
			img = dst.clone();
		}
		imshow("img", imagine);
		imshow("eroziune", dst);
		waitKey();
	}
}

Mat dilatare1(Mat img, int n) {

	int di[8] = { 0, -1, -1, -1,  0 , 1, 1, 1 };
	int dj[8] = { 1,  1,  0, -1, -1, -1, 0, 1 };

	Mat dst = Mat(img.rows, img.cols, CV_8UC1);

	while (n > 0) {

		for (int i = 0; i < img.rows; i++)
			for (int j = 0; j < img.cols; j++)
			{
				if (img.at<uchar>(i, j) == 0) {

					dst.at<uchar>(i, j) = 0;

					for (int k = 0; k < 8; k++) {
						Point2i neighbor(i + di[k], j + dj[k]);

						if (isInside(img, neighbor.x, neighbor.y))
							dst.at<uchar>(neighbor.x, neighbor.y) = 0;
					}
				}

				else {
					dst.at<uchar>(i, j) = 255;
				}
			}

		n--;
		img = dst.clone();

	}

	return dst;

}

Mat eroziune1(Mat img, int n) {

	int di[8] = { 0, -1, -1, -1,  0 , 1, 1, 1 };
	int dj[8] = { 1,  1,  0, -1, -1, -1, 0, 1 };

	Mat dst = Mat(img.rows, img.cols, CV_8UC1);

	while (n > 0) {

		for (int i = 0; i < img.rows; i++)
			for (int j = 0; j < img.cols; j++)
			{

				if (img.at<uchar>(i, j) == 0) {

					bool flag = false;
					for (int k = 0; k < 8 && !flag; k++) {
						Point2i neighbor(i + di[k], j + dj[k]);
						if (!isInside(img, neighbor.x, neighbor.y) || img.at<uchar>(neighbor.x, neighbor.y) != 0) flag = true;
					}


					if (!flag)	dst.at<uchar>(i, j) = 0;

					else {
						dst.at<uchar>(i, j) = 255;
					}

				}

				else {
					dst.at<uchar>(i, j) = 255;
				}
			}
		n--;
		img = dst.clone();
	}

	return dst;
}

void deschidere(int n) {

	char fname[MAX_PATH];

	while (openFileDlg(fname))
	{
		Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat dst = img.clone();

		while (n > 0) {
			dst = eroziune1(img, 1);
			dst = dilatare1(dst, 1);

			n--;
		}
		imshow("dst", dst);
		waitKey();
	}
}

void inchidere(int n) {
	char fname[MAX_PATH];

	while (openFileDlg(fname))
	{
		Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat dst = img.clone();

		while (n > 0) {
			dst = dilatare1(img, 1);
			dst = eroziune1(dst, 1);

			n--;
		}
		imshow("dst", dst);
		waitKey();
	}
}

void contur1() {

	char fname[MAX_PATH];

	while (openFileDlg(fname))
	{
		Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat dst = Mat(img.rows, img.cols, CV_8UC1);

		for (int i = 0; i < dst.rows; i++)
			for (int j = 0; j < dst.cols; j++)
				dst.at<uchar>(i, j) = 255;

		Mat eroziune = eroziune1(img, 1);

		for (int i = 0; i < img.rows; i++) {
			for (int j = 0; j < img.cols; j++) {
				if (img.at<uchar>(i, j) != eroziune.at<uchar>(i, j)) {
					dst.at<uchar>(i, j) = 0;
				}
			}
		}
		imshow("dst", dst);
		waitKey();
	}
}

bool comparareImagini(Mat img1, Mat img2) {
	for (int i = 0; i < img1.rows; i++)
		for (int j = 0; j < img1.cols; j++)
		{
			if (img1.at<uchar>(i, j) != img2.at<uchar>(i, j)) return false;
		}
	return true;
}

void umplere() {

	int di[8] = { 0, -1, -1, -1,  0 , 1, 1, 1 };
	int dj[8] = { 1,  1,  0, -1, -1, -1, 0, 1 };

	char fname[MAX_PATH];

	while (openFileDlg(fname))
	{
		Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat complement = img.clone();
		Mat aux = img.clone();


		// calculam imaginea complementa

		for (int i = 0; i < img.rows; i++)
			for (int j = 0; j < img.cols; j++)
				if (img.at<uchar>(i, j) == 0) complement.at<uchar>(i, j) = 255;
				else complement.at<uchar>(i, j) = 0;

		// aici ar trebui sa avem imaginea finala(aux), facem initializare

		for (int i = 0; i < img.rows; i++)
			for (int j = 0; j < img.cols; j++)
				aux.at<uchar>(i, j) = 255;

		bool flag = false;

		// cautam primul pixel din interiorul unui obiect 

		for (int i = 0; i < img.rows && !flag; i++)
			for (int j = 0; j < img.cols && !flag; j++)
				if (img.at<uchar>(i, j) == 255 && isInside(img, i + 1, j + 1) && img.at<uchar>(i + 1, j + 1) == 255 && isInside(img, i, j - 1) && isInside(img, i - 1, j) && isInside(img, i + 1, j - 1) && isInside(img, i - 1, j + 1) && img.at<uchar>(i, j - 1) == 0 && img.at<uchar>(i - 1, j) == 0 && img.at<uchar>(i + 1, j - 1) == 0 && img.at<uchar>(i - 1, j + 1) == 0) {
					aux.at<uchar>(i, j) = 0;
					flag = true;
					printf("%d %d", i, j);
				}
		if (flag) {

			// aux1 e imaginea rezultata dupa un pas de umplere

			Mat aux1 = Mat(img.rows, img.cols, CV_8UC1);

			for (int i = 0; i < img.rows; i++)
				for (int j = 0; j < img.cols; j++)
					aux1.at<uchar>(i, j) = 255;

			flag = false;

			while (!flag)
			{

				// realizare dilatare

				for (int i = 0; i < img.rows; i++) for (int j = 0; j < img.cols; j++) {
					if (aux.at<uchar>(i, j) == 0)
					{
						for (int k = 0; k < 8; k++)
						{
							if (isInside(aux1, i + di[k], j + dj[k]))
								aux1.at<uchar>(i + di[k], j + dj[k]) = 0;
						}
					}

				}

				//se face intersectia cu complementara imaginii
				for (int i = 0; i < img.rows; i++) for (int j = 0; j < img.cols; j++) {
					if (aux1.at<uchar>(i, j) == 0 && complement.at<uchar>(i, j) == 0)
						aux1.at<uchar>(i, j) = 0;
					else
						aux1.at<uchar>(i, j) = 255;
				}

				//conditie de terminare

				if (comparareImagini(aux, aux1))
					flag = true;

				for (int i = 0; i < img.rows; i++) for (int j = 0; j < img.cols; j++)
				{
					aux.at<uchar>(i, j) = aux1.at<uchar>(i, j);
				}

			}

			// reuniunea cu conturul imaginii

			for (int i = 0; i < aux.rows; i++)
				for (int j = 0; j < aux.cols; j++)

					if (aux.at<uchar>(i, j) == 0 || complement.at<uchar>(i, j) == 255)
						aux.at<uchar>(i, j) = 0;
					else
						aux.at<uchar>(i, j) = 255;


			imshow("img", img);
			imshow("aux", aux);
			imshow("dst", complement);
			waitKey();

		}


		else {
			imshow("img", img);
			imshow("aux", img);
			imshow("dst", complement);
			waitKey();
		}
	}
}


void histogramaa() {

	char fname[MAX_PATH];

	while (openFileDlg(fname))
	{
		Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int M = img.rows*img.cols;
		int hist[256] = { 0 };
		int hc[256] = { 0 };

		float media;
		float suma = 0, suma1 = 0, deviatia;



		for (int i = 0; i < img.rows; i++)
		{
			for (int j = 0; j < img.cols; j++)
			{
				hist[img.at<uchar>(i, j)]++;
				suma = suma + img.at<uchar>(i, j);
			}
		}

		media = suma / M;

		hc[0] = hist[0];

		for (int i = 0; i < img.rows; i++)
		{
			for (int j = 0; j < img.cols; j++)
			{
				suma1 = suma1 + (img.at<uchar>(i, j) - media)*(img.at<uchar>(i, j) - media);
			}
			if (i > 0) hc[i] = hc[i - 1] + hist[i];
		}


		deviatia = sqrt(suma1 / M);

		printf("Media nivelurilor de intensitate este %f\n", media);
		printf("Deviatia nivelurilor de intensitate este %f\n", deviatia);

		showHistogram("Histograma imagine", hist, img.cols, img.rows);
		showHistogram("Histograma cumulativa", hc, img.cols, img.rows);

		waitKey();
	}
}


void binarizare_automata(int prag, float eroare) {

	char fname[MAX_PATH];

	while (openFileDlg(fname))
	{
		Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat dst = Mat(img.rows, img.cols, CV_8UC1);
		int M = img.rows*img.cols;
		int N1, N2, Imin = img.at<uchar>(0, 0), Imax = img.at<uchar>(0, 0);
		float ug1, ug2;
		float suma1 = 0, suma2 = 0;
		int ok = 0;
		int nouPrag = prag;
		int pragInitial = prag;
		int hist[256] = { 0 };

		for (int i = 0; i < img.rows; i++)
		{
			for (int j = 0; j < img.cols; j++)
			{
				hist[img.at<uchar>(i, j)]++;
				if (img.at<uchar>(i, j) < Imin) Imin = img.at<uchar>(i, j);
				if (img.at<uchar>(i, j) > Imax) Imax = img.at<uchar>(i, j);
				if (img.at<uchar>(i, j) != 0 && img.at<uchar>(i, j) != 255) ok = 1;
			}
		}

		if (ok == 1) {

			do {

				pragInitial = nouPrag;

				N1 = 0;
				N2 = 0;
				suma1 = hist[0];
				suma2 = 0;

				if (pragInitial == 255) { nouPrag = 255; break; }


				for (int i = Imin; i < pragInitial + 1; i++) {
					N1 = N1 + hist[i];
					suma1 = suma1 + hist[i] * i;
				}

				for (int i = pragInitial + 1; i <= Imax; i++) {
					N2 = N2 + hist[i];
					suma2 = suma2 + hist[i] * i;
				}

				ug1 = suma1 / N1;
				ug2 = suma2 / N2;

				nouPrag = (int)(ug1 + ug2) / 2;

			}

			while (nouPrag - pragInitial >= eroare);
		}

		else {

			nouPrag = 127;
		}


		for (int i = 0; i < img.rows; i++)
			for (int j = 0; j < img.cols; j++)
				if (img.at<uchar>(i, j) < nouPrag) dst.at<uchar>(i, j) = 0;
				else dst.at<uchar>(i, j) = 255;

		imshow("src", img);
		imshow("dst", dst);

		waitKey();
	}
}

void functii_transformare(int goutmin, int goutmax, float coefGamma, int modificareLuminazitate)
{
	char fname[MAX_PATH];

	while (openFileDlg(fname))
	{
		Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat dst_neg = Mat(img.rows, img.cols, CV_8UC1);
		Mat dst_contrast = Mat(img.rows, img.cols, CV_8UC1);
		Mat dst_corectie_gamma = Mat(img.rows, img.cols, CV_8UC1);
		Mat dst_luminozitate = Mat(img.rows, img.cols, CV_8UC1);

		int hist[256] = { 0 };
		int hist_neg[256] = { 0 };
		int hist_contrast[256] = { 0 };
		int hist_corectie_gamma[256] = { 0 };
		int hist_luminozitate[256] = { 0 };
		int L = 255;

		int ginmin = img.at<uchar>(0, 0), ginmax = img.at<uchar>(0, 0);

		for (int i = 0; i < img.rows; i++)
			for (int j = 0; j < img.cols; j++)
			{
				hist[img.at<uchar>(i, j)]++;

				dst_neg.at<uchar>(i, j) = 255 - img.at<uchar>(i, j);
				hist_neg[dst_neg.at<uchar>(i, j)]++;

				if (L * powf((float)img.at<uchar>(i, j) / L, coefGamma) > 255) dst_corectie_gamma.at<uchar>(i, j) = 255;
				else if (L * powf((float)img.at<uchar>(i, j) / L, coefGamma) < 0) dst_corectie_gamma.at<uchar>(i, j) = 0;
				else dst_corectie_gamma.at<uchar>(i, j) = (uchar)L * powf((float)img.at<uchar>(i, j) / L, coefGamma);
				hist_corectie_gamma[dst_corectie_gamma.at<uchar>(i, j)]++;

				if (img.at<uchar>(i, j) + modificareLuminazitate > 255) dst_luminozitate.at<uchar>(i, j) = 255;
				else if (img.at<uchar>(i, j) + modificareLuminazitate < 0) dst_luminozitate.at<uchar>(i, j) = 0;
				else dst_luminozitate.at<uchar>(i, j) = img.at<uchar>(i, j) + modificareLuminazitate;
				hist_luminozitate[dst_luminozitate.at<uchar>(i, j)]++;

				if (img.at<uchar>(i, j) < ginmin) ginmin = img.at<uchar>(i, j);
				if (img.at<uchar>(i, j) > ginmax) ginmax = img.at<uchar>(i, j);

			}

		if ((float)(goutmax - goutmin) / (ginmax - ginmin) > 1) printf("Latire\n");
		else printf("Ingustare\n");

		if (coefGamma > 1) printf("Decodificare/decomprimare gamma\n");
		else printf("Codificare/comprimare gamma\n");

		for (int i = 0; i < img.rows; i++)
			for (int j = 0; j < img.cols; j++)
			{
				dst_contrast.at<uchar>(i, j) = (uchar)(goutmin + (float)(img.at<uchar>(i, j) - ginmin) * (goutmax - goutmin) / (ginmax - ginmin));
				hist_contrast[dst_contrast.at<uchar>(i, j)]++;
			}

		imshow("src", img);
		imshow("dst-neg", dst_neg);
		imshow("dst-contrast", dst_contrast);
		imshow("dst-corectie-gamma", dst_corectie_gamma);
		imshow("dst-luminozitate", dst_luminozitate);

		showHistogram("Histograma imagine", hist, img.cols, img.rows);
		showHistogram("Histograma imagine negata", hist_neg, img.cols, img.rows);
		showHistogram("Histograma contrast modificat", hist_contrast, img.cols, img.rows);
		showHistogram("Histograma corectie gamma", hist_contrast, img.cols, img.rows);
		showHistogram("Histograma luminozitate", hist_luminozitate, img.cols, img.rows);

		waitKey();

	}
}

void egalizare_histograma() {

	char fname[MAX_PATH];

	while (openFileDlg(fname))
	{
		Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat dst = Mat(img.rows, img.cols, CV_8UC1);

		int hist[256] = { 0 };
		int hist_egalizare[256] = { 0 };
		float prob_hist[256] = { 0.0 };
		float prob_cumulativa_hist[256] = { 0.0 };
		float prob_cumulativa_hist_norm[256] = { 0.0 };
		int L = 255;

		for (int i = 0; i < img.rows; i++)
			for (int j = 0; j < img.cols; j++)
			{
				hist[img.at<uchar>(i, j)]++;

			}
		for (int i = 0; i < 256; i++) {
			prob_hist[i] = (float)hist[i] / (img.rows*img.cols);
			if (i == 0) prob_cumulativa_hist[i] = prob_hist[i];
			else prob_cumulativa_hist[i] = prob_cumulativa_hist[i - 1] + prob_hist[i];
			prob_cumulativa_hist_norm[i] = L * prob_cumulativa_hist[i];
		}

		for (int i = 0; i < img.rows; i++)
			for (int j = 0; j < img.cols; j++)
			{
				if (hist[img.at<uchar>(i, j)] != 0) dst.at<uchar>(i, j) = (uchar)prob_cumulativa_hist_norm[img.at<uchar>(i, j)];
				hist_egalizare[dst.at<uchar>(i, j)]++;
			}
		imshow("src", img);
		imshow("dst", dst);

		showHistogram("Histograma imagine", hist, img.cols, img.rows);
		showHistogram("Histograma egalizata", hist_egalizare, img.cols, img.rows);

		waitKey();
	}
}

bool egalitate_coordonate_centroizi(Vec3b *clusters, Vec3b *clusters1, int k) {
	for (int i = 0; i < k; i++) if (clusters[i][2] != clusters1[i][2] || clusters[i][1] != clusters1[i][1] || clusters[i][0] != clusters1[i][0]) return false;
	return true;
}

Vec3b* maxime_histograma(Mat img, int *rez) {
	int hist[256] = { 0 };
	int hist1[256] = { 0 };
	int hist2[256] = { 0 };
	float fdp[256] = { 0.0 };
	float fdp1[256] = { 0.0 };
	float fdp2[256] = { 0.0 };
	float TH = 0.0003;
	int WH = 5;
	char fname[MAX_PATH];
	float v;
	int praguri[256];
	int praguri1[256];
	int praguri2[256];
	int contor = 1;
	int contor1 = 1;
	int contor2 = 1;
	praguri[0] = 0;
	praguri1[0] = 0;
	praguri2[0] = 0;

	for (int i = 0; i < img.rows; i++) for (int j = 0; j < img.cols; j++) {
		Vec3b pixel = img.at<Vec3b>(i, j);
		hist[pixel[2]]++;
		hist1[pixel[1]]++;
		hist2[pixel[0]]++;
	}

	for (int i = 0; i < 256; i++) {
		fdp[i] = (float)hist[i] / (img.rows*img.cols);
		fdp1[i] = (float)hist1[i] / (img.rows*img.cols);
		fdp2[i] = (float)hist2[i] / (img.rows*img.cols);
	}

	int suma = 0;
	int suma1 = 0;
	int suma2 = 0;
	for (int k = WH; k < 256 - WH; k++) {
		int ok = 0;
		for (int i = k - WH; i <= k + WH; i++) {
			suma = suma + fdp[i];
			if (fdp[k] < fdp[i]) ok = 1;
		}
		v = (float)suma / (2 * WH + 1);
		if (fdp[k] > v + TH && ok == 0) {
			praguri[contor] = k; contor++;
		}
	}

	for (int k = WH; k < 256 - WH; k++) {
		int ok = 0;
		for (int i = k - WH; i <= k + WH; i++) {
			suma1 = suma1 + fdp1[i];
			if (fdp1[k] < fdp1[i]) ok = 1;
		}
		v = (float)suma1 / (2 * WH + 1);
		if (fdp1[k] > v + TH && ok == 0) {
			praguri1[contor1] = k; contor1++;
		}
	}

	for (int k = WH; k < 256 - WH; k++) {
		int ok = 0;
		for (int i = k - WH; i <= k + WH; i++) {
			suma2 = suma2 + fdp2[i];
			if (fdp2[k] < fdp2[i]) ok = 1;
		}
		v = (float)suma2 / (2 * WH + 1);
		if (fdp2[k] > v + TH && ok == 0) {
			praguri2[contor2] = k; contor2++;
		}
	}

	praguri[contor] = 255;
	contor++;
	praguri1[contor1] = 255;
	contor1++;
	praguri2[contor2] = 255;
	contor2++;

	int minim = min(contor, min(contor1, contor2));

	Vec3b *intermediar = (Vec3b *)malloc(sizeof(Vec3b)*(minim - 2));


	Vec3b pixel;

	for (int i = 1; i < minim - 1; i++) {
		pixel[2] = praguri[i];
		pixel[1] = praguri1[i];
		pixel[0] = praguri2[i];
		intermediar[i - 1] = pixel;
	}

	*rez = minim - 2;

	return intermediar;

}


void k_means(int k, int prag, int caz, int hsv) {
	char fname[MAX_PATH];

	while (openFileDlg(fname))
	{
		Mat img = imread(fname, CV_LOAD_IMAGE_COLOR);

		if (hsv) img = split_rgb_to_hsv(img);

		Mat dst = Mat(img.rows, img.cols, CV_8UC3);

		Vec3b *clusters = NULL;
		Vec3b *clusters1 = NULL;

		int *x = NULL;
		int *y = NULL;

		switch (caz) {
		case 1:

			clusters = (Vec3b *)malloc(sizeof(Vec3b)*k);
			clusters1 = (Vec3b *)malloc(sizeof(Vec3b)*k);
			x = (int *)malloc(sizeof(int)*k);
			y = (int *)malloc(sizeof(int)*k);

			for (int i = 0; i < k; i++) {

				x[i] = rand() % img.rows;
				y[i] = rand() % img.cols;
				clusters[i] = img.at<Vec3b>(x[i], y[i]);
				printf("Culorile centroizilor initiali alesi: %d %d %d\n", clusters[i][2], clusters[i][1], clusters[i][0]);
				printf("Coordonate centroizilor initiali: %d %d\n", x[i], y[i]);

			}

			break;

		case 2:
			clusters = (Vec3b *)malloc(sizeof(Vec3b)*k);
			clusters1 = (Vec3b *)malloc(sizeof(Vec3b)*k);
			x = (int *)malloc(sizeof(int)*k);
			y = (int *)malloc(sizeof(int)*k);

			for (int i = 0; i < k; i++) {

			reluare:
				x[i] = rand() % img.rows;
				y[i] = rand() % img.cols;
				clusters[i] = img.at<Vec3b>(x[i], y[i]);
				for (int j = 0; j < i; j++) if (abs(clusters[j][2] - clusters[i][2]) < prag || abs(clusters[j][1] - clusters[i][1]) < prag || abs(clusters[j][0] - clusters[i][0]) < prag) goto reluare;
				printf("Culorile centroizilor initiali alesi: %d %d %d\n", clusters[i][2], clusters[i][1], clusters[i][0]);
				printf("Coordonate centroizilor initiali: %d %d\n", x[i], y[i]);

			}

			break;

		case 3:

			clusters = maxime_histograma(img, &k);
			for (int i = 0; i < k; i++) {
				printf("Culorile centroizilor initiali alesi: %d %d %d\n", clusters[i][2], clusters[i][1], clusters[i][0]);
			}
			clusters1 = (Vec3b *)malloc(sizeof(Vec3b)*k);
			x = (int *)malloc(sizeof(int)*k);
			y = (int *)malloc(sizeof(int)*k);

			break;

		}


		for (int i = 0; i < img.rows; i++)
			for (int j = 0; j < img.cols; j++)
			{
				float dist_min = INT_MAX;
				for (int nr = 0; nr < k; nr++)
				{
					if (i != x[nr] || j != y[nr]) {
						Vec3b pixel = img.at<Vec3b>(i, j);
						float distanta_rgb = 0;
						distanta_rgb = sqrt((clusters[nr][2] - pixel[2])*(clusters[nr][2] - pixel[2]) + (clusters[nr][1] - pixel[1])*(clusters[nr][1] - pixel[1]) + (clusters[nr][0] - pixel[0])*(clusters[nr][0] - pixel[0]));
						if (distanta_rgb < dist_min) {

							dist_min = distanta_rgb;
							dst.at<Vec3b>(i, j) = clusters[nr];
						}
					}

					else {
						dst.at<Vec3b>(i, j) = clusters[nr];
					}



				}
			}

		do {

			for (int i = 0; i < k; i++) { clusters1[i] = clusters[i]; }

			for (int nr = 0; nr < k; nr++) {
				int nr_pixeli = 0;
				int red_sum = 0;
				int blue_sum = 0;
				int green_sum = 0;

				for (int i = 0; i < dst.rows; i++)
					for (int j = 0; j < dst.cols; j++)
					{
						if (dst.at<Vec3b>(i, j) == clusters[nr]) {

							Vec3b pixel = dst.at<Vec3b>(i, j);
							nr_pixeli++;
							red_sum = red_sum + pixel[2];
							green_sum = green_sum + pixel[1];
							blue_sum = blue_sum + pixel[0];
						}
					}
				if (nr_pixeli != 0) {
					clusters[nr][2] = red_sum / nr_pixeli;
					clusters[nr][1] = green_sum / nr_pixeli;
					clusters[nr][0] = blue_sum / nr_pixeli;
				}

				printf("Noile culori ale centroizilor: %d %d %d \n", clusters[nr][2], clusters[nr][1], clusters[nr][0]);
			}

			for (int i = 0; i < img.rows; i++)
				for (int j = 0; j < img.cols; j++)
				{
					float dist_min = INT_MAX;
					for (int nr = 0; nr < k; nr++)
					{
						if (i != x[nr] || j != y[nr]) {
							Vec3b pixel = img.at<Vec3b>(i, j);
							float distanta_rgb = 0;
							distanta_rgb = sqrt((clusters[nr][2] - pixel[2])*(clusters[nr][2] - pixel[2]) + (clusters[nr][1] - pixel[1])*(clusters[nr][1] - pixel[1]) + (clusters[nr][0] - pixel[0])*(clusters[nr][0] - pixel[0]));
							if (distanta_rgb < dist_min) {

								dist_min = distanta_rgb;
								dst.at<Vec3b>(i, j) = clusters[nr];
							}
						}

						else {
							dst.at<Vec3b>(i, j) = clusters[nr];
						}



					}
				}

		}

		while (!egalitate_coordonate_centroizi(clusters, clusters1, k));

		imshow("src", img);
		imshow("dst", dst);
		waitKey();
	}

}

void filtru_trece_jos(Mat_<float> H)
{
	char fname[MAX_PATH];

	while (openFileDlg(fname))
	{
		Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat dst = Mat(img.rows, img.cols, CV_8UC1);

		for (int i = 0; i < dst.rows; i++)
			for (int j = 0; j < dst.cols; j++) dst.at<uchar>(i, j) = 0;

		int sum_matrice = 0;

		int k = (H.rows - 1) / 2;

		for (int i = 0; i < H.rows; i++)
			for (int j = 0; j < H.cols; j++) sum_matrice = sum_matrice + H(i, j);


		for (int i = k; i < img.rows - k; i++)
			for (int j = k; j < img.cols - k; j++)
			{
				int suma_partiala = 0;

				for (int u = 0; u < H.rows; u++) {
					for (int v = 0; v < H.cols; v++)
					{
						suma_partiala = suma_partiala + H(u, v) * img.at<uchar>(i + u - k, j + v - k);

					}


				}
				dst.at<uchar>(i, j) = suma_partiala / sum_matrice;

			}



		imshow("src", img);
		imshow("dst", dst);
		waitKey();
	}

}

void filtru_trece_sus(Mat_<float> H) {

	char fname[MAX_PATH];

	while (openFileDlg(fname))
	{
		Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat dst = Mat(img.rows, img.cols, CV_32FC1);
		Mat dst1 = Mat(img.rows, img.cols, CV_8UC1);

		for (int i = 0; i < dst.rows; i++)
			for (int j = 0; j < dst.cols; j++) dst.at<float>(i, j) = 0;


		int k = (H.rows - 1) / 2;


		for (int i = k; i < img.rows - k; i++)
			for (int j = k; j < img.cols - k; j++)
			{
				int suma_partiala = 0;

				for (int u = 0; u < H.rows; u++) {
					for (int v = 0; v < H.cols; v++)
					{
						suma_partiala = suma_partiala + H(u, v) * img.at<uchar>(i + u - k, j + v - k);

					}


				}
				dst.at<float>(i, j) = suma_partiala;
			}

		int minim = INT_MAX;
		int maxim = INT_MIN;

		for (int i = 0; i < dst.rows; i++)
			for (int j = 0; j < dst.cols; j++)
			{
				if (dst.at<float>(i, j) > maxim) maxim = dst.at<float>(i, j);
				if (dst.at<float>(i, j) < minim)  minim = dst.at<float>(i, j);
			}

		int L = 255;

		for (int i = 0; i < dst1.rows; i++)
			for (int j = 0; j < dst1.cols; j++)
			{

				dst1.at<uchar>(i, j) = (uchar)L * (dst.at<float>(i, j) - minim) / (maxim - minim);

			}


		imshow("src", img);
		imshow("dst", dst1);
		waitKey();
	}
}

void sortare(int v[], int n) {
	bool flag = false;

	while (!flag) {
		flag = true;
		for (int i = 0; i < n - 1; i++) {
			if (v[i] > v[i + 1]) {
				int t = v[i];
				v[i] = v[i + 1];
				v[i + 1] = t;
				flag = false;
			}
		}
	}
}


void filtru_median_minim_maxim(int w) {

	char fname[MAX_PATH];

	int * di = (int *)malloc(sizeof(int)*(w*w));
	int * dj = (int *)malloc(sizeof(int)*(w*w));

	int value = (w - 1) / 2 * (-1);
	int value1 = (w - 1) / 2 * (-1);

	for (int i = 0; i < w*w; i = i + 1) {


		if (i%w == 0 && i != 0) {
			value++;
			value1 = (w - 1) / 2 * (-1);
		}

		di[i] = value;
		dj[i] = value1;

		value1++;

	}

	while (openFileDlg(fname))
	{
		double t = (double)getTickCount();

		Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat dst = Mat(img.rows, img.cols, CV_8UC1);
		Mat dst1 = Mat(img.rows, img.cols, CV_8UC1);
		Mat dst2 = Mat(img.rows, img.cols, CV_8UC1);

		for (int i = 0; i < img.rows; i++)
			for (int j = 0; j < img.cols; j++)
			{
				dst.at<uchar>(i, j) = 0;
				dst1.at<uchar>(i, j) = 0;
				dst2.at<uchar>(i, j) = 0;

			}

		int k = (w - 1) / 2;


		for (int i = k; i < img.rows - k; i++)
			for (int j = k; j < img.cols - k; j++)
			{
				int * numbers = (int *)malloc(sizeof(int)*(w*w));
				for (int nr = 0; nr < w*w; nr++) numbers[nr] = img.at<uchar>(i + di[nr], j + dj[nr]);
				sortare(numbers, w*w);

				dst.at<uchar>(i, j) = numbers[w * w / 2 + 1];
				dst1.at<uchar>(i, j) = numbers[0];
				dst2.at<uchar>(i, j) = numbers[w*w - 1];

			}

		t = ((double)getTickCount() - t) / getTickFrequency();
		// Afișarea la consolă a timpului de procesare [ms]
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("src", img);
		imshow("Filtru median", dst);
		imshow("Filtru minim", dst1);
		imshow("Filtru maxim", dst2);
		waitKey();

	}
}

void filtru_gaussian(int w) {

	Mat g = Mat(w, w, CV_32FC1);
	float delta = (float)w / 6;

	int x0 = (w - 1) / 2;
	int y0 = (w - 1) / 2;

	for (int i = 0; i < g.rows; i++)
		for (int j = 0; j < g.cols; j++)
		{
			g.at<float>(i, j) = exp(double((i - x0)*(i - x0) + (j - y0)*(j - y0)) / (2 * delta*delta) *(-1)) / (2 * PI * delta * delta);
		}

	printf("Nucleul gaussian:\n");

	for (int i = 0; i < g.rows; i++) {
		for (int j = 0; j < g.cols; j++) {
			printf("%f ", g.at<float>(i, j));
		}
		printf("\n");
	}

	char fname[MAX_PATH];

	while (openFileDlg(fname))
	{
		double t = (double)getTickCount();

		Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat dst = Mat(img.rows, img.cols, CV_32FC1);
		Mat dst1 = Mat(img.rows, img.cols, CV_8UC1);

		for (int i = 0; i < img.rows; i++)
			for (int j = 0; j < img.cols; j++)
				dst.at<float>(i, j) = 0;

		int k = (w - 1) / 2;

		for (int i = k; i < img.rows - k; i++)
			for (int j = k; j < img.cols - k; j++)
			{
				float suma_partiala = 0;

				for (int u = 0; u < g.rows; u++) {
					for (int v = 0; v < g.cols; v++)
					{
						suma_partiala = suma_partiala + g.at<float>(u, v) * img.at<uchar>(i + u - k, j + v - k);

					}
				}
				dst.at<float>(i, j) = suma_partiala;

			}

		float minim = INT_MAX;
		float maxim = INT_MIN;

		for (int i = 0; i < dst.rows; i++)
			for (int j = 0; j < dst.cols; j++)
			{
				if (dst.at<float>(i, j) > maxim) maxim = dst.at<float>(i, j);
				if (dst.at<float>(i, j) < minim)  minim = dst.at<float>(i, j);
			}

		int L = 255;

		for (int i = 0; i < dst.rows; i++)
			for (int j = 0; j < dst.cols; j++)
			{

				dst1.at<uchar>(i, j) = (uchar)L * (dst.at<float>(i, j) - minim) / (maxim - minim);

			}


		t = ((double)getTickCount() - t) / getTickFrequency();
		// Afișarea la consolă a timpului de procesare [ms]
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("src", img);
		imshow("dst", dst1);
		waitKey();
	}
}

Mat gaussian_filter(int w, Mat img) {

	Mat g = Mat(w, w, CV_32FC1);
	float delta = (float)w / 6;

	int x0 = (w - 1) / 2;
	int y0 = (w - 1) / 2;

	for (int i = 0; i < g.rows; i++)
		for (int j = 0; j < g.cols; j++)
		{
			g.at<float>(i, j) = exp(double((i - x0)*(i - x0) + (j - y0)*(j - y0)) / (2 * delta*delta) *(-1)) / (2 * PI * delta * delta);
		}

	Mat dst = Mat(img.rows, img.cols, CV_32FC1);
	Mat dst1 = Mat(img.rows, img.cols, CV_8UC1);

	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
			dst.at<float>(i, j) = 0;

	int k = (w - 1) / 2;

	for (int i = k; i < img.rows - k; i++)
		for (int j = k; j < img.cols - k; j++)
		{
			float suma_partiala = 0;

			for (int u = 0; u < g.rows; u++) {
				for (int v = 0; v < g.cols; v++)
				{
					suma_partiala = suma_partiala + g.at<float>(u, v) * img.at<uchar>(i + u - k, j + v - k);

				}
			}
			dst.at<float>(i, j) = suma_partiala;

		}

	float minim = INT_MAX;
	float maxim = INT_MIN;

	for (int i = 0; i < dst.rows; i++)
		for (int j = 0; j < dst.cols; j++)
		{
			if (dst.at<float>(i, j) > maxim) maxim = dst.at<float>(i, j);
			if (dst.at<float>(i, j) < minim)  minim = dst.at<float>(i, j);
		}

	int L = 255;

	for (int i = 0; i < dst.rows; i++)
		for (int j = 0; j < dst.cols; j++)
		{

			dst1.at<uchar>(i, j) = (uchar)L * (dst.at<float>(i, j) - minim) / (maxim - minim);

		}

	return dst1;
}


void conv_lab11(Mat_<float> H, Mat_<float> H1) {

	char fname[MAX_PATH];

	while (openFileDlg(fname))
	{
		Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat dst = Mat(img.rows, img.cols, CV_32FC1);
		Mat dst1 = Mat(img.rows, img.cols, CV_32FC1);
		Mat dst2 = Mat(img.rows, img.cols, CV_8UC1);
		Mat dst3 = Mat(img.rows, img.cols, CV_8UC1);
		Mat Mag = Mat(img.rows, img.cols, CV_32FC1);
		Mat Mag1 = Mat(img.rows, img.cols, CV_32FC1);
		Mat dst4 = Mat(img.rows, img.cols, CV_8UC1);
		Mat unghi = Mat(img.rows, img.cols, CV_32FC1);
		Mat dst5 = Mat(img.rows, img.cols, CV_8UC1);
		Mat suprimare = Mat(img.rows, img.cols, CV_8UC1);
		Mat binarizareAdaptiva = Mat(img.rows, img.cols, CV_8UC1);

		imshow("src", img);

		img = gaussian_filter(5, img);

		for (int i = 0; i < dst.rows; i++)
			for (int j = 0; j < dst.cols; j++) {
				dst.at<float>(i, j) = 0;
				Mag1.at<float>(i, j) = 0;
			}

		int k = (H.rows - 1) / 2;


		for (int i = k; i < img.rows - k; i++)
			for (int j = k; j < img.cols - k; j++)
			{
				int suma_partiala = 0;
				int suma_partiala1 = 0;

				for (int u = 0; u < H.rows; u++) {
					for (int v = 0; v < H.cols; v++)
					{
						suma_partiala = suma_partiala + H(u, v) * img.at<uchar>(i + u - k, j + v - k);
						suma_partiala1 = suma_partiala1 + H1(u, v) * img.at<uchar>(i + u - k, j + v - k);

					}


				}
				dst.at<float>(i, j) = suma_partiala;
				dst1.at<float>(i, j) = suma_partiala1;
				Mag.at<float>(i, j) = sqrt(dst.at<float>(i, j)*dst.at<float>(i, j) + dst1.at<float>(i, j) * dst1.at<float>(i, j));
				unghi.at<float>(i, j) = atan2((double)dst1.at<float>(i, j), (double)dst.at<float>(i, j));
			}

		int minim = INT_MAX;
		int maxim = INT_MIN;
		int minim1 = INT_MAX;
		int maxim1 = INT_MIN;
		int minim2 = INT_MAX;
		int maxim2 = INT_MIN;
		int minim3 = INT_MAX;
		int maxim3 = INT_MIN;

		for (int i = 0; i < dst.rows; i++)
			for (int j = 0; j < dst.cols; j++)
			{
				if (dst.at<float>(i, j) > maxim) maxim = dst.at<float>(i, j);
				if (dst.at<float>(i, j) < minim)  minim = dst.at<float>(i, j);

				if (dst1.at<float>(i, j) > maxim1) maxim1 = dst1.at<float>(i, j);
				if (dst1.at<float>(i, j) < minim1)  minim1 = dst1.at<float>(i, j);

				if (Mag.at<float>(i, j) > maxim2) maxim2 = Mag.at<float>(i, j);
				if (Mag.at<float>(i, j) < minim2)  minim2 = Mag.at<float>(i, j);

				if (unghi.at<float>(i, j) > maxim3) maxim3 = unghi.at<float>(i, j);
				if (unghi.at<float>(i, j) < minim3)  minim3 = unghi.at<float>(i, j);
			}

		int L = 255;

		for (int i = 0; i < dst.rows; i++)
			for (int j = 0; j < dst.cols; j++)
			{

				dst2.at<uchar>(i, j) = (uchar)L * (dst.at<float>(i, j) - minim) / (maxim - minim);
				dst3.at<uchar>(i, j) = (uchar)L * (dst1.at<float>(i, j) - minim1) / (maxim1 - minim1);
				dst4.at<uchar>(i, j) = (uchar)L * (Mag.at<float>(i, j) - minim2) / (maxim2 - minim2);
				dst5.at<uchar>(i, j) = (uchar)L * (unghi.at<float>(i, j) - minim3) / (maxim3 - minim3);

			}

		for (int i = 0; i < dst.rows; i++)
			for (int j = 0; j < dst.cols; j++) {
				unghi.at<float>(i, j) = 180 * unghi.at<float>(i, j) / PI;
				if (unghi.at<float>(i, j) < 0) unghi.at<float>(i, j) += 360;
			}

		int di1 = 0, dj1 = 0, di2 = 0, dj2 = 0;

		for (int i = 1; i < img.rows - 1; i++)
			for (int j = 1; j < img.cols - 1; j++)
			{


				if ((unghi.at<float>(i, j) >= 22.5 &&unghi.at<float>(i, j) <= 67.5) || (unghi.at<float>(i, j) >= 202.5 && unghi.at<float>(i, j) <= 247.5)) {
					di1 = 1;
					dj1 = -1;
					di2 = -1;
					dj2 = 1;
				}
				if ((unghi.at<float>(i, j) > 67.5 && unghi.at<float>(i, j) <= 112.5) || (unghi.at<float>(i, j) > 247.5 && unghi.at<float>(i, j) <= 292.5)) {
					di1 = -1;
					dj1 = 0;
					di2 = 1;
					dj2 = 0;
				}

				if ((unghi.at<float>(i, j) > 112.5 && unghi.at<float>(i, j) <= 157.5) || (unghi.at<float>(i, j) > 292.5 && unghi.at<float>(i, j) <= 337.5)) {
					di1 = -1;
					dj1 = -1;
					di2 = 1;
					dj2 = 1;
				}

				if ((unghi.at<float>(i, j) >= 0 && unghi.at<float>(i, j) < 22.5) || (unghi.at<float>(i, j) > 157.5 && unghi.at<float>(i, j) < 202.5) || unghi.at<float>(i, j) > 337.5) {
					di1 = 0;
					dj1 = -1;
					di2 = 0;
					dj2 = 1;
				}


				if (Mag.at<float>(i, j) >= Mag.at<float>(i + di1, j + dj1) && Mag.at<float>(i, j) >= Mag.at<float>(i + di2, j + dj2)) {
					Mag1.at<float>(i, j) = Mag.at<float>(i, j);
				}
				else Mag1.at<float>(i, j) = 0;

			}


		for (int i = 0; i < dst.rows; i++)
			for (int j = 0; j < dst.cols; j++)
			{
				suprimare.at<uchar>(i, j) = (uchar)(Mag1.at<float>(i, j)) / (4 * sqrt(2));

			}

		float p = 0.1;

		int hist[256] = { 0 };

		for (int i = 0; i < suprimare.rows; i++)
			for (int j = 0; j < suprimare.cols; j++)
			{
				hist[suprimare.at<uchar>(i, j)]++;
			}


		int nrNonMuchie = (1 - p) * (suprimare.rows * suprimare.cols - hist[0]);

		int prag = 1;
		int sum = hist[1];
		int count = 1;
		while (sum < nrNonMuchie) {
			count++;
			sum = sum + hist[count];
			prag = count;
		}

		for (int i = 0; i < binarizareAdaptiva.rows; i++)
			for (int j = 0; j < binarizareAdaptiva.cols; j++) {

				if (suprimare.at<uchar>(i, j) < prag) {
					binarizareAdaptiva.at<uchar>(i, j) = 0;
				}
				else {
					binarizareAdaptiva.at<uchar>(i, j) = 255;
				}

			}

		int pragInalt = prag;
		int pragCoborat = 0.4 * pragInalt;

		imshow("Imaginea dupa suprimarea non-maximelor magnitudinii", suprimare);

		for (int i = 0; i < suprimare.rows; i++)
			for (int j = 0; j < suprimare.cols; j++)
			{
				if (suprimare.at<uchar>(i, j) > pragInalt) suprimare.at<uchar>(i, j) = 255;
				else if (suprimare.at<uchar>(i, j) >= pragCoborat && suprimare.at<uchar>(i, j) <= pragInalt) suprimare.at<uchar>(i, j) = 128;
				else suprimare.at<uchar>(i, j) = 0;

			}

		imshow("Initializare muchii tari si slabe", suprimare);

		for (int i = 0; i < suprimare.rows; i++)
			for (int j = 0; j < suprimare.cols; j++) {

				if (suprimare.at<uchar>(i, j) == 255) {

					int di[8] = { 0, -1, -1, -1,  0 , 1, 1, 1 };
					int dj[8] = { 1,  1,  0, -1, -1, -1, 0, 1 };

					queue<Point2i> Q;
					Q.push(Point2i(i, j));

					while (!Q.empty()) {
						Point2i point = Q.front();
						Q.pop();
						for (int nr = 0; nr < 8; nr++)
						{
							if (isInside(suprimare, point.x + di[nr], point.y + dj[nr]) && suprimare.at<uchar>(point.x + di[nr], point.y + dj[nr]) == 128) {
								suprimare.at<uchar>(point.x + di[nr], point.y + dj[nr]) = 255;
								Q.push(Point2d(point.x + di[nr], point.y + dj[nr]));
							}
						}
					}

				}
			}

		for (int i = 0; i < suprimare.rows; i++)
			for (int j = 0; j < suprimare.cols; j++) {
				if (suprimare.at<uchar>(i, j) == 128) suprimare.at<uchar>(i, j) = 0;
			}


		imshow("Imaginea dupa filtrarea gaussiana", img);
		imshow("Fx", dst2);
		imshow("Fy", dst3);
		imshow("Magnitudine", dst4);
		imshow("Unghi", dst5);
		imshow("Prelungirea muchiilor prin histereza", suprimare);
		imshow("BinarizareAdaptiva", binarizareAdaptiva);

		waitKey();
	}
}


void exemplu() {
	char fname[MAX_PATH];

	while (openFileDlg(fname))
	{
		Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat dst = Mat(img.rows, img.cols, CV_8UC1);

		int hist[256] = { 0 };
		int maxim = -1;

		for (int i = 0; i < img.rows; i++)
			for (int j = 0; j < img.cols; j++)
			{
				hist[img.at<uchar>(i, j)]++;
				if (img.at<uchar>(i, j) > maxim) maxim = img.at<uchar>(i, j);

			}
		printf("Histograma\n");
		for (int i = 0; i < 256; i++) printf("%d ", hist[i]);
		printf("\n");
		printf("Maxim : %d\n", maxim);

		for (int i = 0; i < img.rows; i++)
			for (int j = 0; j < img.cols; j++)
				if (img.at<uchar>(i, j) < maxim) dst.at<uchar>(i, j) = 0;
				else dst.at<uchar>(i, j) = 255;

		imshow("src", img);
		imshow("dst", dst);
		waitKey();
	}
}

Mat generare_culori1(Mat img, Mat labels,Vec3b **culorile,int *n) {

	Mat dst = Mat(img.rows, img.cols, CV_8UC3);

	int maxLabel = INT_MIN;

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {

			if (labels.at<int>(i, j) > maxLabel)
				maxLabel = labels.at<int>(i, j);
		}
	}

	Vec3b* culori = (Vec3b *)malloc(sizeof(Vec3b)*(maxLabel+1));

	for (int k = 0; k < maxLabel; k++) {

		reluare:

		uchar r = d(gen);
		uchar g = d(gen);
		uchar b = d(gen);
		culori[k] = Vec3b(r, g, b);
		for (int i = 0; i < k; i++) if (culori[i] == culori[k]) goto reluare;
	}
	
	culori[0] = Vec3b(255, 0, 0);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {

			if (labels.at<int>(i, j) >= 0) {

				dst.at<Vec3b>(i, j) = culori[labels.at<int>(i, j)];

			}

			else
				dst.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
		}
	}
	*culorile = culori;
	*n = maxLabel;

	return dst;
}

void exemplu1() {


	char fname[MAX_PATH];

	while (openFileDlg(fname))
	{
		Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat dst = Mat(img.rows, img.cols, CV_8UC3);
		dst.setTo(255);
		int label = 0;
		Mat labels = Mat(img.rows, img.cols, CV_32SC1);

		for (int i = 0; i < labels.rows; i++) for (int j = 0; j < labels.cols; j++) labels.at<int>(i, j) = 0;

		int di[8] = { 0, -1, -1, -1,  0 , 1, 1, 1 };
		int dj[8] = { 1,  1,  0, -1, -1, -1, 0, 1 };

		for (int i = 0; i < img.rows; i++) {
			for (int j = 0; j < img.cols; j++) {

				if (img.at<uchar>(i, j) == 0 && labels.at<int>(i, j) == 0) {

					label++;
					queue<Point2i> Q;
					labels.at<int>(i, j) = label;
					Q.push({ i,j });

					while (!Q.empty()) {

						Point2i q = Q.front(); Q.pop();

						for (int k = 0; k < 8; k++) {

							Point2i neighbor(q.x + di[k], q.y + dj[k]);

							if (isInside(img, neighbor.x, neighbor.y) && img.at<uchar>(neighbor.x, neighbor.y) == 0 && labels.at<int>(neighbor.x, neighbor.y) == 0) {

								labels.at<int>(neighbor.x, neighbor.y) = label;
								Q.push(neighbor);
							}
						}
					}



				}
			}

		}

		Vec3b* culori;
		int dimensiune;

		Mat img1 = generare_culori1(img, labels,&culori,&dimensiune);
		printf("Dimensiune %d\n", dimensiune);
		
		for (int i = 0; i < dimensiune; i++) printf("Culoare %d %d %d \n", culori[i][2], culori[i][1], culori[i][0]);

		imshow("as", img1);

		int numar = 0;

		for (int k = 0; k < dimensiune; k++) {
			float aria = 0;
			float perimetru = 0;
			for(int i=0;i<img1.rows;i++)
				for (int j = 0; j < img1.cols; j++) {
					if (img1.at<Vec3b>(i, j) == culori[k]) {
						aria++;
						if (img1.at<Vec3b>(i, j) == culori[k] && (img1.at<Vec3b>(i + 1, j) != culori[k] || img1.at<Vec3b>(i - 1, j) != culori[k] || img1.at<Vec3b>(i, j + 1) != culori[k] || img1.at<Vec3b>(i, j - 1) != culori[k] || img1.at<Vec3b>(i + 1, j + 1) != culori[k] || img1.at<Vec3b>(i + 1, j - 1) != culori[k] || img1.at<Vec3b>(i - 1, j + 1) != culori[k] || img1.at<Vec3b>(i - 1, j - 1) != culori[k])) {
							perimetru++;
						}
					}
				}

			float alungire = 4 * PI * aria / (perimetru * perimetru);
			printf("Arie: %f \n", aria);
			printf("Perimetru: %f\n", perimetru);
			printf("Alungire: %f\n", alungire);
			

			if (alungire > 0.6) {
				numar++;
				for (int i = 0; i < img1.rows; i++)
					for (int j = 0; j < img1.cols; j++) {
						if (img1.at<Vec3b>(i, j) == culori[k]) {
							dst.at<Vec3b>(i, j) = img1.at<Vec3b>(i, j);
						}
					}
			}
		}

		printf("\nAvem %d obiecte rotunde", numar);
		imshow("src", img);
		imshow("dst", dst);
		waitKey();
	}

}

int main()
{
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Image negative - diblook style\n");
		printf(" 4 - BGR->HSV\n");
		printf(" 5 - Resize image\n");
		printf(" 6 - Canny edge detection\n");
		printf(" 7 - Edges in a video sequence\n");
		printf(" 8 - Snap frame from live video\n");
		printf(" 9 - Mouse callback demo\n");
		printf(" 10 - Imagine\n");
		printf(" 11 - Gray image\n");
		printf(" 12 - Aduna val\n");
		printf(" 13 - Scade val\n");
		printf(" 14 - Inmultire val\n");
		printf(" 15 - Inversa\n");
		printf(" 16 - Negative image\n");
		printf(" 17 - Color to gray 1\n");
		printf(" 18 - Binarizare 1\n");
		printf(" 19 - Split RGB\n");
		printf(" 20 - Split RGB Color\n");
		printf(" 21 - Split RGB Color to HSV\n");
		printf(" 22 - Verificare apartinere imagine\n");
		printf(" 23 - Histograma\n");
		printf(" 24 - Histograma 3 canale\n");
		printf(" 25 - FDP - calcul + afisare histograma\n");
		printf(" 26 - Histograma cu numar redus de acumulatoare\n");
		printf(" 27 - Determinare praguri multiple\n");
		printf(" 28 - Arie, centru de masa, axa de alungire, perimetru, axa de subtiere si elongatia.\n");
		printf(" 29 - Eliminare obiecte care au aria mai mica decat o valoare citita si alungirea intr-un anumit interval citit de la tastatura\n");
		printf(" 30 - Etichetare cu traversare in latime\n");
		printf(" 31 - Etichetare cu traversare in adancime\n");
		printf(" 32 - Etichetare cu doua treceri\n");
		printf(" 33 - Contorul unei imagini\n");
		printf(" 34 - Reconstructie imagine\n");
		printf(" 35 - Dilatare imagine\n");
		printf(" 36 - Eroziune imagine\n");
		printf(" 37 - Deschidere imagine\n");
		printf(" 38 - Inchidere imagine\n");
		printf(" 39 - Contur cu operatii morfologice\n");
		printf(" 40 - Umplere\n");
		printf(" 41 - Media, deviatia, histograma normal si cea cumulativa\n");
		printf(" 42 - Binarizare automata\n");
		printf(" 43 - Functii transformare\n");
		printf(" 44 - Egalizare histograma\n");
		printf(" 45 - Algoritmul K-Means\n");
		printf(" 46 - Filtru trece jos\n");
		printf(" 47 - Filtru trece sus\n");
		printf(" 48 - Filtru median,minim si maxim\n");
		printf(" 49 - Filtru gaussian\n");
		printf(" 50 - Metoda lui Canny\n");
		printf(" 51 - Exemplu exercitiu colocviu\n");
		printf(" 52 - Exemplu exercitiu colocviu 1\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");

		float b[25] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
		Mat_<float> H1(5, 5, b);

		float a[9] = { 1, 1, 1, 1, 1, 1, 1, 1, 1 };
		Mat_<float> H(3, 3, a);

		float c[9] = { 0, -1, 0, -1, 4, -1, 0, -1, 0 };
		Mat_<float> H2(3, 3, c);

		float d[9] = { -1, -1, -1, -1, 8, -1, -1, -1, -1 };
		Mat_<float> H3(3, 3, d);

		float e[9] = { 0, -1, 0, -1, 5, -1, 0, -1, 0 };
		Mat_<float> H4(3, 3, e);

		float f[9] = { -1, -1, -1, -1, 9, -1, -1, -1, -1 };
		Mat_<float> H5(3, 3, f);

		float sobx[9] = { -1, 0, 1, -1, 0, 1, -1, 0, 1 };
		Mat_<float>  sobelx(3, 3, sobx);

		float soby[9] = { 1, 1, 1, 0, 0, 0, -1, -1, -1 };
		Mat_<float>  sobely(3, 3, soby);

		bool hsv = false;

		scanf("%d", &op);
		switch (op)
		{
		case 1:
			testOpenImage();
			break;
		case 2:
			testOpenImagesFld();
			break;
		case 3:
			testParcurgereSimplaDiblookStyle(); //diblook style
			break;
		case 4:
			//testColor2Gray();
			testBGR2HSV();
			break;
		case 5:
			testResize();
			break;
		case 6:
			testCanny();
			break;
		case 7:
			testVideoSequence();
			break;
		case 8:
			testSnap();
			break;
		case 9:
			testMouseClick();
			break;
		case 10:
			imagine();
			break;
		case 11:
			grayImage();
			break;
		case 12:
			aduna();
			break;
		case 13:
			scade();
			break;
		case 14:
			inmultire();
			break;
		case 15:
			inversa();
			break;
		case 16:
			negative_image();
			break;
		case 17:
			color2gray1();
			break;
		case 18:
			binarizare();
			break;
		case 19:
			split_rgb();
			break;
		case 20:
			split_rgb_color();
			break;
		case 21:
			split_rgb_to_hsv();
			break;
		case 22:
			verificare();
			break;
		case 23:
			histograma();
			break;
		case 24:
			histograma_3_canale();
			break;
		case 25:
			fdp();
			break;
		case 26:
			printf("Citeste numarul de acumulatoare:\n");
			int n;
			scanf("%d", &n);

			histacumulatoare(n);
		case 27:
			determinare_praguri_multiple();
			break;
		case 28:
			lab4();
			break;
		case 29:
			printf("Citeste arie si intervalul orientarii dupa care se fac decuparea obiectelor:\n");
			float arie, phimin, phimax;
			scanf("%f", &arie);
			scanf("%f", &phimin);
			scanf("%f", &phimax);
			ariemaimica(arie, phimin, phimax);
			break;
		case 30:
			BFS();
			break;
		case 31:
			DFS();
			break;
		case 32:
			parcurgere_dubla();
			break;
		case 33:
			contur();
			break;
		case 34:
			reconstructie();
			break;
		case 35:
			printf("Numar de repetitii: ");
			int nr;
			scanf("%d", &nr);
			dilatare(nr);
			break;
		case 36:
			printf("Numar de repetitii: ");
			scanf("%d", &nr);
			eroziune(nr);
			break;
		case 37:
			printf("Numar de repetitii: ");
			scanf("%d", &nr);
			deschidere(1);
			break;
		case 38:
			printf("Numar de repetitii: ");
			scanf("%d", &nr);
			inchidere(1);
			break;
		case 39:
			contur1();
			break;
		case 40:
			umplere();
			break;
		case 41:
			histogramaa();
			break;
		case 42:
			int prag;
			float eroare;
			printf("Citire prag(valoare intreaga) si eroare(valoare reala):\n");
			scanf("%d", &prag);
			scanf("%f", &eroare);
			binarizare_automata(prag, eroare);
			break;
		case 43:
			int goutmin;
			int goutmax;
			float coefGamma;
			int valoareCrestere;
			printf("Citire intensitate minima si maxima pentru iesire(intre 0-255), coeficientul gamma si valoarea de crestere a intensitatii:\n");
			scanf("%d", &goutmin);
			scanf("%d", &goutmax);
			scanf("%f", &coefGamma);
			scanf("%d", &valoareCrestere);
			functii_transformare(goutmin, goutmax, coefGamma, valoareCrestere);
			break;
		case 44:
			egalizare_histograma();
			break;
		case 45:

			int optiune;
			int k;
			while (true) {
				printf("Transformare imagine din spatiul RGB in HSV? (1/0-Da/Nu)\n");
				scanf("%d", &optiune);
				if (optiune == 1) hsv = true;
				else hsv = false;


				printf("Alege optiune:\n");
				printf("1. Alege in mod aleator punctele de pornire.\n");
				printf("2. Alege punctele din imagine astfel incat acestea sa fie distincte (sa aiba culoarea diferita).\n");
				printf("3. Selectare puncte de pornire in functie de continutul imaginii (selectie automata).\n");
				printf("4. Inchidere aplicatie.\n");

				scanf("%d", &optiune);

				if (optiune == 1) {
					printf("Citire k (nr. de zone in care vrei sa fie segmentata imaginea:\n");
					scanf("%d", &k);
					k_means(k, 0, optiune, hsv);
				}
				else if (optiune == 2) {
					printf("Citire k (nr. de zone in care vrei sa fie segmentata imaginea:\n");
					scanf("%d", &k);
					printf("Citire prag pentru diferentele de culoare intre fiecare zona:\n");
					scanf("%d", &prag);
					k_means(k, prag, optiune, hsv);

				}
				else if (optiune == 3) {
					k_means(0, 0, 3, hsv);
				}

				else if (optiune == 4) {
					return 0;
				}
			}
			break;

		case 46:

			//filtru_trece_jos(H);
			filtru_trece_jos(H1);
			break;

		case 47:

			//filtru_trece_sus(H2);
			//filtru_trece_sus(H3);
			//filtru_trece_sus(H4);
			filtru_trece_sus(H5);
			break;

		case 48:
			printf("Citire valoare w:\n");
			int w;
			scanf("%d", &w);
			filtru_median_minim_maxim(w);
			break;

		case 49:
			printf("Citire valoare w:\n");
			scanf("%d", &w);
			filtru_gaussian(w);
			break;

		case 50:
			conv_lab11(sobelx, sobely);
			break;

		case 51:
			exemplu();
			break;

		case 52:
			exemplu1();
			break;
		}
	} while (op != 0);
	return 0;
}