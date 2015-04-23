##########################
#
# Miscellaneous R Tools
#
###########################

##########################
#
# generic outlier function
# Written by Martin Corley
# September 09, 2009
# http://psy-ed.wikidot.com/datamanipulation
# License: Creative Commons Attribution-ShareAlike 3.0 License
#          http://creativecommons.org/licenses/by-sa/3.0/
#
###########################

outliers <- function(x, index=NULL, sds=2.5) {
  if (is.data.frame(x)) {
    as.data.frame(sapply(x, outliers, index, sds))
  } else if (is.matrix(x)) {
    apply(x, 2, outliers, index, sds)
  } else if (is.list(x)) {
    lapply(x, outliers, index, sds)
  } else if (is.vector(x)) {
    if (!is.null(index)) {
      if (!is.list(index)) {
        index <- list(index) # make sure index is a list
      }
      unsplit(outliers(split(x,index),index=NULL,sds=sds),index)
    } else {
      bound <- sds*sd(x,na.rm=T)
      m <- mean(x,na.rm=T)
      (abs(x-m) > bound)
    }
  } else {
    cat("outliers not implemented for class ",class(x),"\n",sep="")
  }
}


##########################
#
# vif stepwise reduction function
# Written by Marcus W. Beck
# February 05, 2013
# http://beckmw.wordpress.com/2013/02/05/collinearity-and-stepwise-vif-selection/
# License: Creative Commons Attribution-ShareAlike 3.0 License
#          http://creativecommons.org/licenses/by-sa/3.0/
#
###########################

#stepwise VIF function used below
vif_func<-function(in_frame,thresh=10,trace=T){

  require(fmsb)
  
  if(class(in_frame) != 'data.frame') in_frame<-data.frame(in_frame)
  #get initial vif value for all comparisons of variables
  vif_init<-NULL
  for(val in names(in_frame)){
    form_in<-formula(paste(val,' ~ .'))
    vif_init<-rbind(vif_init,c(val,VIF(lm(form_in,data=in_frame))))
  }
  vif_max<-max(as.numeric(vif_init[,2]))

  if(vif_max < thresh){
    if(trace==T){ #print output of each iteration
      prmatrix(vif_init,collab=c('var','vif'),rowlab=rep('',nrow(vif_init)),quote=F)
      cat('\n')
      cat(paste('All variables have VIF < ', thresh,', max VIF ',round(vif_max,2), sep=''),'\n\n')
    }
    return(names(in_frame))
  }
  else{

    in_dat<-in_frame

    #backwards selection of explanatory variables, stops when all VIF values are below 'thresh'
    while(vif_max >= thresh){
      vif_vals<-NULL

      for(val in names(in_dat)){
        form_in<-formula(paste(val,' ~ .'))
        vif_add<-VIF(lm(form_in,data=in_dat))
        vif_vals<-rbind(vif_vals,c(val,vif_add))
      }
      max_row<-which(vif_vals[,2] == max(as.numeric(vif_vals[,2])))[1]

      vif_max<-as.numeric(vif_vals[max_row,2])

      if(vif_max<thresh) break
      if(trace==T){ #print output of each iteration
        prmatrix(vif_vals,collab=c('var','vif'),rowlab=rep('',nrow(vif_vals)),quote=F)
        cat('\n')
        cat('removed: ',vif_vals[max_row,1],vif_max,'\n\n')
        flush.console()
      }

      in_dat<-in_dat[,!names(in_dat) %in% vif_vals[max_row,1]]

    }

    return(names(in_dat))
  }
}