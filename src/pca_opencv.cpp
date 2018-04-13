#include <iostream>
#include <fstream>
#include <string>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <fstream>
#include <sstream>


static void read_imgList(const std::string& filename, std::vector<cv::Mat>& images) 
{
	std::ifstream file(filename.c_str(), std::ifstream::in);
	if (!file) 
	{
		std::string error_message = "No valid input file was given, please check the given filename.";
		CV_Error(cv::Error::StsBadArg, error_message);
	}
	
	std::string line;
	while (getline(file, line)) 
	{
		images.push_back(cv::imread(line, cv::IMREAD_GRAYSCALE));
	}
}


static cv::Mat formatImagesForPCA(const std::vector<cv::Mat> &data)
{
	auto data_rows = data[0].rows;
	auto data_cols = data[0].cols;
	auto data_0_size = data[0].size;
	auto data_size = data.size();

	cv::Mat dst(static_cast<int>(data.size()), data[0].rows*data[0].cols, CV_32F);
	for (unsigned int i = 0; i < data.size(); i++)
	{
		cv::Mat image_row = data[i].clone().reshape(1, 1);
		cv::Mat row_i = dst.row(i);
		image_row.convertTo(row_i, CV_32F);
	}
	return dst;
}


static cv::Mat toGrayscale(cv::InputArray _src) 
{
	cv::Mat src = _src.getMat();
	// only allow one channel
	if (src.channels() != 1) 
	{
		CV_Error(cv::Error::StsBadArg, "Only Matrices with one channel are supported");
	}
	// create and return normalized image
	cv::Mat dst;
	cv::normalize(_src, dst, 0, 255, cv::NORM_MINMAX, CV_8UC1);
	return dst;
}

struct params
{
	cv::Mat data;
	int ch;
	int rows;
	cv::PCA pca;
	std::string winName;
};


static void onTrackbar(int pos, void* ptr)
{
	pos = std::max(5, pos);
	
	std::cout << "Retained Variance = " << pos << "%   ";
	std::cout << "re-calculating PCA..." << std::flush;
	
	double var = pos / 100.0;
	struct params *p = (struct params *)ptr;
	p->pca = cv::PCA(p->data, cv::Mat(), cv::PCA::DATA_AS_ROW, var);
	cv::Mat point = p->pca.project(p->data.row(0));
	cv::Mat reconstruction = p->pca.backProject(point);
	reconstruction = reconstruction.reshape(p->ch, p->rows);
	reconstruction = toGrayscale(reconstruction);
	cv::imshow(p->winName, reconstruction);
	std::cout << "done!   # of principal components: " << p->pca.eigenvectors.rows << std::endl;
}

// Main
int main(int argc, char** argv)
{
	cv::CommandLineParser parser(argc, argv, "{@input||image list}{help h||show help message}");
	if (parser.has("help"))
	{
		parser.printMessage();
		exit(0);
	}

	//// Get the path to your CSV.
	std::string imgList = parser.get<std::string>("@input");
	if (imgList.empty())
	{
		parser.printMessage();
		exit(1);
	}


	// vector to hold the images
	std::vector<cv::Mat> images;
	// Read in the data. This can fail if not valid
	try 
	{
		read_imgList(imgList, images);
	}
	catch (cv::Exception& e) 
	{
		std::cerr << "Error opening file \"" << imgList << "\". Reason: " << e.msg << std::endl;
		exit(1);
	}

	// Quit if there are not enough images for this demo.
	if (images.size() <= 1) 
	{
		std::string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
		CV_Error(cv::Error::StsError, error_message);
	}

	// Reshape and stack images into a rowMatrix
	cv::Mat data = formatImagesForPCA(images);
	// perform PCA
	cv::PCA pca(data, cv::Mat(), cv::PCA::DATA_AS_ROW, 0.95); // trackbar is initially set here, also this is a common value for retainedVariance
													  // Demonstration of the effect of retainedVariance on the first image
	cv::Mat point = pca.project(data.row(0)); // project into the eigenspace, thus the image becomes a "point"
	cv::Mat reconstruction = pca.backProject(point); // re-create the image from the "point"
	reconstruction = reconstruction.reshape(images[0].channels(), images[0].rows); // reshape from a row vector into image shape
	reconstruction = toGrayscale(reconstruction); // re-scale for displaying purposes
	
	cv::Mat mean = pca.mean.clone();
	mean = mean.reshape(images[0].channels(), images[0].rows); // reshape from a row vector into image shape
	mean = toGrayscale(mean); // re-scale for displaying purposes
	
	cv::imwrite("../data/heads/heads_pca_mean_cv.jpg", mean);
	cv::imwrite("../data/heads/heads_pca_mean_rec.jpg", reconstruction);

	// init highgui window
	std::string winName = "Reconstruction | press 'q' to quit";
	cv::namedWindow(winName, cv::WINDOW_NORMAL);
	// params struct to pass to the trackbar handler
	params p;
	p.data = data;
	p.ch = images[0].channels();
	p.rows = images[0].rows;
	p.pca = pca;
	p.winName = winName;
	
	// create the tracbar
	int pos = 95;
	cv::createTrackbar("Retained Variance (%)", winName, &pos, 100, onTrackbar, (void*)&p);
	// display until user presses q
	imshow(winName, reconstruction);
	char key = 0;
	while (key != 'q')
		key = (char)cv::waitKey();
	return 0;
}