###############
#load dataset #
###############
### Kaggle Scripts: Ponpare Coupon Purchase Prediction ###
### Original Author: Subhajit Mandal ###
### Some score improving changes by Fred H Seymour
rm(list=ls())
cat('\014')
dir <- '~/dropbox/data science/kaggle/coupon purchase prediction/'
cat("Reading data\n")
cpdtr <- read.csv(paste0(dir,"coupon_detail_train.csv"))
cpltr <- read.csv(paste0(dir,"coupon_list_train.csv"))
cplte <- read.csv(paste0(dir,"coupon_list_test.csv"))
ulist <- read.csv(paste0(dir,"user_list.csv"))
cpvtr <- read.csv(paste0(dir,"coupon_visit_train.csv"),nrows=-1) # big file
cpvtr <- cpvtr[cpvtr$PURCHASE_FLG!=1,c("VIEW_COUPON_ID_hash","USER_ID_hash")]

#####################
#preprocessing      #
#####################
# --------- VALIDPERIOD ----------
# fix cpltr$VALIDPERIOD error where valid should be VALIDEND minu VALIDFROM plus one!
# put into factor of 0s for NAs and 1s actual values
cpltr$VALIDPERIOD[is.na(cpltr$VALIDPERIOD)] <- -1
cpltr$VALIDPERIOD <- cpltr$VALIDPERIOD+1
cpltr$VALIDPERIOD[cpltr$VALIDPERIOD>0] <- 1
cpltr$VALIDPERIOD <- as.factor(cpltr$VALIDPERIOD)

cplte$VALIDPERIOD[is.na(cplte$VALIDPERIOD)] <- -1
cplte$VALIDPERIOD <- cplte$VALIDPERIOD+1
cplte$VALIDPERIOD[cplte$VALIDPERIOD>0] <- 1
cplte$VALIDPERIOD <- as.factor(cplte$VALIDPERIOD)

# ------------- USEABLE_DATEs ------------
# sets up sum of coupon USABLE_DATEs for training and test dataset
for (i in 12:20) {
  cpltr[is.na(cpltr[,i]),i] <- 0;    cpltr[cpltr[,i]>1,i] <- 1
  cplte[is.na(cplte[,i]),i] <- 0;    cplte[cplte[,i]>1,i] <- 1
}
cpltr$USABLE_DATE_sum <- rowSums(cpltr[,12:18])
cplte$USABLE_DATE_sum <- rowSums(cplte[,12:18])
cpltr$USABLE_DATE_hol <- rowSums(cpltr[,19:20])
cplte$USABLE_DATE_hol <- rowSums(cplte[,19:20])

################################
#combine training and test sets#
################################
# start train set by merging coupon_detail_train and coupon_list_train
# to get USER_ID_hash by coupon
train <- merge(cpdtr,cpltr)
train <- train[,c("COUPON_ID_hash","USER_ID_hash","GENRE_NAME","DISCOUNT_PRICE","DISPPERIOD",
                  "large_area_name","small_area_name","VALIDPERIOD","USABLE_DATE_sum","USABLE_DATE_hol",
                  "ken_name","PRICE_RATE","CAPSULE_TEXT")]
# append test set to the training set for model.matrix factor column conversion
cplte$USER_ID_hash <- "dummyuser"
cpchar <- cplte[,c("COUPON_ID_hash","USER_ID_hash","GENRE_NAME","DISCOUNT_PRICE","DISPPERIOD",
                   "large_area_name","small_area_name","VALIDPERIOD","USABLE_DATE_sum","USABLE_DATE_hol",
                   "ken_name","PRICE_RATE","CAPSULE_TEXT")]
train <- rbind(train,cpchar)

# NA imputation to values of 1
train[is.na(train)] <- 1
# feature engineering
train$DISCOUNT_PRICE <- 1/log10(train$DISCOUNT_PRICE)    
train$DISPPERIOD[train$DISPPERIOD > 7] <- 7; 
train$DISPPERIOD <- train$DISPPERIOD / 7
train$USABLE_DATE_sum <- train$USABLE_DATE_sum / 7
train$USABLE_DATE_hol <- train$USABLE_DATE_hol / 2

# convert the factors to columns of 0's and 1's
train <- cbind(train[,c(1,2)],model.matrix(~ -1 + .,train[,-c(1,2)],
                                           contrasts.arg=lapply(train[,names(which(sapply(train[,-c(1,2)], is.factor)==TRUE))], 
                                                                contrasts, contrasts=FALSE)))
# --------- separate the test from train ------------- 
# following factor column conversion
test <- train[train$USER_ID_hash=="dummyuser",]
test <- test[,-2]
train <- train[train$USER_ID_hash!="dummyuser",]

# Numeric attributes cosine multiplication factors set to 1
train$DISCOUNT_PRICE <- 1
train$DISPPERIOD <- 1
train$USABLE_DATE_sum <- 1

# Create starting uchar for all users initialized to zero
uchar <- data.frame(USER_ID_hash=ulist[,"USER_ID_hash"])
uchar <- cbind(uchar,matrix(0, nrow=dim(uchar)[1], ncol=(dim(train)[2] -2)))
names(uchar) <- names(train)[2:dim(train)[2]]

# Incorporate the purchase training data from train, use sum function    
uchar <- aggregate(.~USER_ID_hash, data=rbind(uchar,train[,-1]),FUN=sum)

# -------------- Add visit training data in chunks due to large dataset -------------
imax <- dim(cpvtr)[1]   
i2 <- 1
while (i2 < imax) {  # this loop takes a few minutes      
  i1 <- i2
  i2 <- i1 + 100000
  if (i2 > imax) i2 <- imax
  cat("Merging coupon visit data i1=",i1," i2=",i2,"\n")
  trainv <- merge(cpvtr[i1:i2,],cpltr, by.x="VIEW_COUPON_ID_hash", by.y="COUPON_ID_hash")
  trainv <- trainv[,c("VIEW_COUPON_ID_hash","USER_ID_hash","GENRE_NAME","DISCOUNT_PRICE","DISPPERIOD",                          
                      "large_area_name","small_area_name","VALIDPERIOD","USABLE_DATE_sum","USABLE_DATE_hol",
                      "ken_name","PRICE_RATE","CAPSULE_TEXT")]
  #same treatment as with coupon_detail train data
  trainv$DISCOUNT_PRICE <- 1
  train$DISPPERIOD      <- 1
  train$USABLE_DATE_sum <- 1
  train$USABLE_DATE_hol <- 1
  trainv[is.na(trainv)] <- 1
  trainv <- cbind(trainv[,c(1,2)],model.matrix(~ -1 + .,trainv[,-c(1,2)],
                                               contrasts.arg=lapply(trainv[,names(which(sapply(trainv[,-c(1,2)], is.factor)==TRUE))], 
                                                                    contrasts, contrasts=FALSE)))
  # discount coupon visits relative to coupon purchases
  couponVisitFactor <- .005      
  trainv[,3:dim(trainv)[2]] <- trainv[,3:dim(trainv)[2]] * couponVisitFactor  
  uchar <- aggregate(.~USER_ID_hash, data=rbind(uchar,trainv[,-1]),FUN=sum)    
}

train$PRICE_RATE <- (train$PRICE_RATE*train$PRICE_RATE)/(100*100)


#############
#save files #
#############
save(file = paste0(dir,"uchar.Rdata"), uchar)
save(file = paste0(dir,"test.Rdata") , test)
save(file = paste0(dir,"ulist.Rdata"), ulist)
save(file = paste0(dir,"trainv.Rdata"), trainv)
save(file = paste0(dir,"train.Rdata"), train)
save(file = paste0(dir,"cpchar.Rdata"), cpchar)
save(file = paste0(dir,"cpdtr.Rdata"), cpdtr)
save(file = paste0(dir,"cpltr.Rdata"), cpltr)
save(file = paste0(dir,"cpvtr.Rdata"), cpvtr)

