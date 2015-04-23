####################################
#
# regression-utils.R
# Written by Austin F. Frank
# February 24, 2011
# https://github.com/aufrank
#
####################################

c. <- function (x) scale(x, scale = FALSE)
z. <- function (x) scale(x)
r. <- function (formula, ...) rstandard(lm(formula, ...))
l. <- function (x) log(x)
s. <- function (x) {
    ## Seber 1977 page 216, from http://dx.doi.org/10.1021/ie970236k
    ## Transforms continuous variable to the range [-1, 1]
    ## In linked paper, recommended before computing orthogonal
    ## polynomials
    (2 * x - max(x) - min(x)) / (max(x) - min(x))
}
p. <- function (x, ...) poly(x, ...)
