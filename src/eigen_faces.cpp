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

// Normalizes a given image into a value range between 0 and 255.
cv::Mat norm_0_255(const cv::Mat& src) {
	// Create and return normalized image:
	cv::Mat dst;
	switch (src.channels()) {
	case 1:
		cv::normalize(src, dst, 0, 255, cv::NORM_MINMAX, CV_8UC1);
		break;
	case 3:
		cv::normalize(src, dst, 0, 255, cv::NORM_MINMAX, CV_8UC3);
		break;
	default:
		src.copyTo(dst);
		break;
	}
	return dst;
}

// Converts the images given in src into a row cv::Matrix.
cv::Mat asRowMatrix(const std::vector<cv::Mat>& src, int rtype_out, double alpha = 1, double beta = 0) {
	// Number of samples:
	size_t n = src.size();
	// Return empty cv::Matrix if no matrices given:
	if (n == 0)
		return cv::Mat();
	// dimensionality of (reshaped) samples
	size_t d = src[0].total();
	std::cout << "src[0]        : " << src[0].rows << ' ' << src[0].cols << ' ' << src[0].channels() << std::endl;
	// Create resulting data matrix:
	cv::Mat data(n, d, rtype_out);

	int channels = (rtype_out < 7) ? 1 : 3;

	// Now copy data:
	for (int i = 0; i < n; i++) 
	{
		//
		if (src[i].empty()) 
		{
			std::string error_message = cv::format("Image number %d was empty, please check your input data.", i);
			CV_Error(CV_StsBadArg, error_message);
		}
		// Make sure data can be reshaped, throw a meaningful exception if not!
		if (src[i].total() != d) 
		{
			std::string error_message = cv::format("Wrong number of elements in matrix #%d! Expected %d was %d.", i, d, src[i].total());
			CV_Error(CV_StsBadArg, error_message);
		}
		// Get a hold of the current row:
		cv::Mat xi = data.row(i);
		// Make reshape happy by cloning for non-continuous matrices:
		if (src[i].isContinuous()) 
		{
			src[i].reshape(channels, 1).convertTo(xi, rtype_out, alpha, beta);
			std::cout << "src           : " << src[i].rows << ' ' << src[i].cols << ' ' << src[i].channels() << std::endl;
		}
		else 
		{
			src[i].clone().reshape(channels, 1).convertTo(xi, rtype_out, alpha, beta);
		}
	}
	return data;
}

int main(int argc, const char *argv[]) 
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

	int rtype = images[0].type();

	// Build a matrix with the observations in row:
	cv::Mat data = asRowMatrix(images, CV_32FC1);
	std::cout << "data          : " << data.rows << ' ' << data.cols << ' ' << data.channels() << std::endl;

	// Number of components to keep for the PCA:
	int num_components = 10;

	// Perform a PCA:
	cv::PCA pca(data, cv::Mat(), CV_PCA_DATA_AS_ROW, num_components);

	// And copy the PCA results:
	cv::Mat mean = pca.mean.clone();
	cv::Mat eigenvalues = pca.eigenvalues.clone();
	cv::Mat eigenvectors = pca.eigenvectors.clone();

	std::cout << "mean          : " << mean.rows << ' ' << mean.cols << ' ' << mean.channels() << std::endl;
	std::cout << "eigenvalues   : " << eigenvalues.rows << ' ' << eigenvalues.cols << ' ' << eigenvalues.channels() << std::endl;
	std::cout << "eigenvectors  : " << eigenvectors.rows << ' ' << eigenvectors.cols << ' ' << eigenvectors.channels() << std::endl;

	cv::namedWindow("avg", cv::WINDOW_NORMAL);
	cv::namedWindow("pc1", cv::WINDOW_NORMAL);
	cv::namedWindow("pc2", cv::WINDOW_NORMAL);
	cv::namedWindow("pc3", cv::WINDOW_NORMAL);

	mean = norm_0_255(mean.reshape(1, images[0].rows));

	// The mean face:
	cv::imshow("avg", mean);
	// The first three eigenfaces:
	cv::imshow("pc1", norm_0_255(pca.eigenvectors.row(0)).reshape(1, images[0].rows));
	cv::imshow("pc2", norm_0_255(pca.eigenvectors.row(1)).reshape(1, images[0].rows));
	cv::imshow("pc3", norm_0_255(pca.eigenvectors.row(2)).reshape(1, images[0].rows));

	cv::imwrite("../data/heads/heads_pca_mean.jpg", mean);


	// Show the images:
	cv::waitKey(0);



	// Success!
	return 0;
}