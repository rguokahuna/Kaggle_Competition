rm(list = ls())
cat('\014')
##############
#load dataset#
##############
dir <- '~/dropbox/data science/kaggle/coupon purchase prediction/'
cplte <- read.csv("~/dropbox/data science/kaggle/coupon purchase prediction/coupon_list_test.csv")
load(file = paste0(dir,'/uchar.Rdata'))
load(file = paste0(dir,"/test.Rdata"))
load(file = paste0(dir,"/ulist.Rdata"))
load(file = paste0(dir,"/cpltr.Rdata"))

##########################
#more feature engineering#
##########################

# -------- modify ken_name with PREF_NAME ---------
uchar <- merge(uchar,ulist[,c('USER_ID_hash','PREF_NAME')])
PREF_NAME <- model.matrix( ~ 0 + ., uchar['PREF_NAME'], contrasts.arg = lapply(uchar['PREF_NAME'],contrasts,contrasts=FALSE))[]
PREF_NAME <- PREF_NAME[,2:ncol(PREF_NAME)]
uchar[uchar[,'PREF_NAME'] != '',85:131] <- PREF_NAME[uchar[,'PREF_NAME'] != '',]
remove(PREF_NAME)

# --------- modify large area name --------------
# get the ken_name
large_area <- matrix(0,dim(uchar)[1],9)
for (user in 1:dim(uchar)[1]) {
  if (uchar[user,'PREF_NAME'] != '') {
    large_area[user, unique(cpltr[cpltr[,'ken_name'] %in% uchar[user,'PREF_NAME'] , 'large_area_name'])] <- 1
  } 
}
uchar[uchar[,'PREF_NAME'] != '',17:25] <- large_area[uchar[,'PREF_NAME'] != '',]
uchar <- uchar[, -which(names(uchar) == 'PREF_NAME') ] # remove PREF_NAME
remove(large_area, user)

# ----------- price rate transformation ------------
test['PRICE_RATE']  <- (test$PRICE_RATE*test$PRICE_RATE)/(100*100)

# -------------- catalog price -----------
uchar['CATALOG_PRICE'] <- 1
test['CATALOG_PRICE']  <- cplte$CATALOG_PRICE/100000


##########################
# Weight Matrix          #
##########################
# GENRE_NAME(13), DISCOUNT_PRICE(1), DISPPERIOD(1), large_area_name(9), small_area_name(55), 
# VALIDPERIOD(2), USABLE_DATE_sum(1), USABLE_DATE_hol(1), ken_name(47), CAPSULE_TEXT(25), PRICE_RATE(1)
# ------- for men -------------------------------------------------------
# genre: Gift card 2, Food 3, Other coupon 4, Hotel and Japanese hotel 8, Leisure 10
genre_m   <- c(1.8,rep(2,3),1.8,1.8,1.8,rep(2,1),1.8,rep(2,1),1.8,1.8,1.8) 
# capsule: イベント 2(3/2.5), ギフトカード 4(1200:800), グルメ 5(2200:1500), ゲストハウス 6(3.5:1)
# capsule: その他 7(2300:1800), ペンション 12(35:15), ホテル 13(800:560), レジャー 15(700:500)
# capsule: 宅配 20(4000:4000), 旅館 21(400:250), 民宿 22(20:10)
capsule_m <- rep(.02,25)
Wm <- diag(c(genre_m, 1.25, 1.25, rep(5.5,9), rep(4.5,55)#over
             ,.625, 3.0, .25,  0.0, rep(1.0,47), rep(-.01,1), capsule_m, rep(.03,1)))
# ----------- for women --------------------------------------------------
# genre: Spa 1, Nail and eye salon 5, Beauty 6, Hair salon 7, Relaxation 9, Health and medical 12, Lesson 11
genre_f <- c(2.1,rep(1.5,3),rep(2.1,3),rep(1.65,1),2.1,rep(1.65),rep(2.1,2),1.65)
# capsule: サービス 1 (60:40), エステ 3 (220:40), ネイル・アイ 8(220:10), ビューティ 9(.2:0)
# capsule: ビューティー 10(15:2), ヘアサロン 11(430:100), リラクゼーション 14(250:125)
# capsule: レッスン 16(300:125), ロッジ17(3:2), 健康・医療 18(22:12), 公共の宿 19(1:.8)
# capsule: 宅配 20(4000:4000), 貸別荘 23(5:3), 通信講座 24(1.2:.8), 通学レッスン 25(6:1)
capsule_f <- c(rep(.02,2),.1,rep(.02,4),.1,.02,.1,.1,rep(.02,13),.1)
Wf <- diag(c(genre_f, 0.75, 1.65, rep(7,9), rep(4.5,55)#over
             ,.425, 2.0, .25, -0.3, rep(1.5,47), rep(-.01,1), capsule_f, rep(.03,1)))

# ----------- calculation of cosine similairties of users and coupons -------------
#calculation of cosine similarities of users and coupons
score = as.matrix(uchar[,2:ncol(uchar)]) %*% Wm %*% t(as.matrix(test[,2:ncol(test)]))
score[ulist$SEX_ID=='f',] = as.matrix(uchar[ulist$SEX_ID=='f',2:ncol(uchar)]) %*% Wf %*% t(as.matrix(test[,2:ncol(test)]))


################
# add distance #
################
# -------------- add distance ---------------
load(file <- paste0(dir,'/distance_test.Rdata'))
distance_test <- 1 / (distance + 1) 
remove(distance)

distance_weight <- .1 * c(rep(1,2), -1.5, -1.5, rep(1,1), rep(1,6), 1, -2)
for (genre in 1:13) {
  # which coupons are in this genre
  distance_test[,cplte[,'GENRE_NAME'] == levels(cpltr[,'GENRE_NAME'])[genre]] <- distance_weight[genre] * distance_test[,cplte[,'GENRE_NAME'] == levels(cpltr[,'GENRE_NAME'])[genre]]
}
score <- score + distance_test


# #####################
# #load orignial uchar#
# #####################
# load(file = paste0(dir,'/uchar.Rdata'))
# uchar['CATALOG_PRICE'] <- 1
# score_o = as.matrix(uchar[,2:ncol(uchar)]) %*% Wm %*% t(as.matrix(test[,2:ncol(test)]))
# score_o[ulist$SEX_ID=='f',] = as.matrix(uchar[ulist$SEX_ID=='f',2:ncol(uchar)]) %*% Wf %*% t(as.matrix(test[,2:ncol(test)]))
# score_o <- score_o + distance_test
# 
# # -------- find coupon with delivery genre --------
# coupon_delivery <- cplte[,'GENRE_NAME'] == '宅配'
# score <- t(apply(score,1,FUN =scale))
# score_o <- t(apply(score_o,1,FUN=scale))
# score <- .8*score + .2*score_o
####################
#colleting results #
####################
#order the list of coupons according to similairties and take only first 10 coupons
uchar$PURCHASED_COUPONS <- do.call(rbind, lapply(1:nrow(uchar),FUN=function(i){
  purchased_cp <- paste(test$COUPON_ID_hash[order(score[i,], decreasing = TRUE)][1:10],collapse=" ")
  return(purchased_cp)
}))
#make submission
submission <- uchar[,c("USER_ID_hash","PURCHASED_COUPONS")]
submission$PURCHASED_COUPONS[rowSums(score)==0] <- ""
write.csv(submission, file=paste0(dir,"/submit.csv"), row.names=FALSE)
