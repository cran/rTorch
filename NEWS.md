# rTorch 0.0.3
* 20190802
* Released to CRAN at 15:20


# rTorch 0.0.2.9000
* 20190802
* Returned from CRAN with notes
* Fix single quotes in DESCRIPTION
* Change `\dontrun` by `\donttest` where applicable
* Get rid of a warning message
* Replace print/cat by message/warning
* Add `\value` to all functions with `@return`
* Added `cran-comments.md`


# rTorch 0.0.2
* July 31 2019
* Submitted to CRAN at 15:30. Received. Waiting manual inspection.
* Test with `appveyor.yml`
* Created repository [r-appveyor](https://github.com/f0nzie/r-appveyor) at Oil Gains Analytics GitHub account. `appveyor` scripts now are called from this repo. Original source is at [krlmlr/r-appveyor](https://github.com/krlmlr/r-appveyor)
* Test with `.travis.yml`
* Copy three functions from reticulate to customize it and be able to specify the conda channel. Using `pytorch` channel in `reticulate.R`.
* Specify torch-cpu and torchvision-cpu in `install.R`
* Move out vignettes to reduce testing time. Will ship separately using `rsuite`.

# rTorch 0.0.1.9013
* July 26 2019
* Vignettes temporarily moved to inst/vignettes to reduce build time of package
* Add function remainder for tensors. Equivalent to `a %% b`
* Change unit tests in `test_generics.R` to use new function `expect_true_tensor`
* Enhance functions `any` and `all`. Add examples
* Add roxygen documentation to two tensor operations
* Change download folders for MNIST datasets under project folder


# rTorch 0.0.1.9012
* July 24 2019
* Change MNIST download folder to ~/raw_data instead of inst/
* On vignette `mnist_fashion_inference.Rmd`:
* Add dropout class to reduce overfitting
* Add a training loop for the dropout class
* Added/remove experimental code to replicate the Python function to visualize the image along with the bar plot. Unsuccessful because R (image) and Python image (plt.imshow) functions use different array dimensions.

# rTorch 0.0.1.9011
* July 22 2019
* Added vignette `mnist_fashion_inference.Rmd`.
* Added vignette `simple_linear_regression.Rmd`.
* Add generic ! (logical not) 
* Fix generics any, all using as_tensor() instead of tensor()


# rTorch 0.0.1.9010
* July 22 2019
* New vignette using PyTorch builtin functions and classes. Rainfall dataset: `linear_regression_rainfall_builtins.Rmd`
* Add comments to `linear_regression_rainfall.Rmd`

# rTorch 0.0.1.9009
* July 22 2019
* Fix version numbers. Missing the number one.

# rTorch 0.0.1.9008
* July 22 2019
* Refresh pkgdown
* Export html files for pkgdown. Modify .gitignore.

# rTorch 0.0.1.9006
* July 22 2019
* Add pkgdown website

# rTorch 0.0.1.9005
* July 22 2019
* Add vignette `png_images_minist_digits.Rmd`. It uses PBG images in a local folder instead of downloading MNIST idx format images.
* Add logical operators to README.

# rTorch 0.0.1.9004
* July 22 2019
* Add vignette `idx_images_minist_digits.Rmd`

# rTorch 0.0.1.9003
* July 21 2019
* New vignette `two_layer_neural_network.Rmd`. Had some problem with the tensor types. Fixed by using shorter generic version of the tensor gradient operation.

# rTorch 0.0.1.9002
* July 21 2019
* Add two more vignettes.
* Get rid of a warning on roxygen documentation
* Remove old code from generics.R

# rTorch 0.0.1.9001
* July 21 2019
* Adding first example as a vignette.
* import Python torch with `py_run_string("import torch")`

# rTorch 0.0.1
* July 21 2019
* alpha version
* first release to Github
* package coming after publication of `rpystats-apollo11`
* still examples to be added


# rTorch 0.0.0.9000

* Added a `NEWS.md` file to track changes to the package.
