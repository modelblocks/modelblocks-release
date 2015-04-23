####################################
#
# mer-utils.R
# Written by Austin F. Frank
# February 24, 2011
# https://github.com/aufrank
#
####################################

vif.mer <- function (fit) {
    ## adapted from rms::vif
    
    v <- vcov(fit)
    nam <- names(fixef(fit))

    ## exclude intercepts
    ns <- sum(1 * (nam == "Intercept" | nam == "(Intercept)"))
    if (ns > 0) {
        v <- v[-(1:ns), -(1:ns), drop = FALSE]
        nam <- nam[-(1:ns)]
    }
    
    d <- diag(v)^0.5
    v <- diag(solve(v/(d %o% d)))
    names(v) <- nam
    v
}

kappa.mer <- function (fit,
                       scale = TRUE, center = FALSE,
                       add.intercept = TRUE,
                       exact = FALSE) {
    X <- fit@X
    nam <- names(fixef(fit))
    
    ## exclude intercepts
    nrp <- sum(1 * (nam == "(Intercept)"))
    if (nrp > 0) {
        X <- X[, -(1:nrp), drop = FALSE]
        nam <- nam[-(1:nrp)]
    }

    if (add.intercept) {
        X <- cbind(rep(1), scale(X, scale = scale, center = center))
        kappa(X, exact = exact)
    } else {
        kappa(scale(X, scale = scale, center = scale), exact = exact)
    }
}

colldiag.mer <- function (fit,
                          scale = TRUE, center = FALSE,
                          add.intercept = TRUE) {
    ## adapted from perturb::colldiag, method in Belsley, Kuh, and
    ## Welsch (1980).  look for a high condition index (> 30) with
    ## more than one high variance propotion.  see ?colldiag for more
    ## tips.
    result <- NULL
    if (center) 
        add.intercept <- FALSE
    if (is.matrix(fit) || is.data.frame(fit)) {
        X <- as.matrix(fit)
        nms <- colnames(fit)
    }
    else if (class(fit) == "mer") {
        nms <- names(fixef(fit))
        X <- fit@X
        if (any(grepl("(Intercept)", nms))) {
            add.intercept <- FALSE
        }
    }
    X <- X[!is.na(apply(X, 1, all)), ]

    if (add.intercept) {
        X <- cbind(1, X)
        colnames(X)[1] <- "(Intercept)"
    }
    X <- scale(X, scale = scale, center = center)

    svdX <- svd(X)
    svdX$d
    condindx <- max(svdX$d)/svdX$d
    dim(condindx) <- c(length(condindx), 1)

    Phi = svdX$v %*% diag(1/svdX$d)
    Phi <- t(Phi^2)
    pi <- prop.table(Phi, 2)
    colnames(condindx) <- "cond.index"
    if (!is.null(nms)) {
        rownames(condindx) <- nms
        colnames(pi) <- nms
        rownames(pi) <- nms
    } else {
        rownames(condindx) <- 1:length(condindx)
        colnames(pi) <- 1:ncol(pi)
        rownames(pi) <- 1:nrow(pi)
    }         

    result <- data.frame(cbind(condindx, pi))
    zapsmall(result)
}

maxcorr.mer <- function (fit,
                         exclude.intercept = TRUE) {
    so <- summary(fit)
    corF <- so@vcov@factors$correlation
    nam <- names(fixef(fit))

    ## exclude intercepts
    ns <- sum(1 * (nam == "Intercept" | nam == "(Intercept)"))
    if (ns > 0 & exclude.intercept) {
        corF <- corF[-(1:ns), -(1:ns), drop = FALSE]
        nam <- nam[-(1:ns)]
    }
    corF[!lower.tri(corF)] <- 0
    maxCor <- max(corF)
    minCor <- min(corF)
    if (abs(maxCor) > abs(minCor)) {
        zapsmall(maxCor)
    } else {
        zapsmall(minCor)
    }
}
