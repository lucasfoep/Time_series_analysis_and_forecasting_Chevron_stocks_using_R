library('ggplot2') # Graphs
library('ggfortify') # Graphs
library('stats') # acf, pacf, box.test
library('forecast') # Arima, forecast, auto.arima and autoplot
library('lmtest') # coeftest
library('fUnitRoots') # adfTest
library('tseries') # garch
library(fGarch) # garchFit
library(rugarch) # ugarch
#library('aTSA') # kpss.test
source('eacf.R')
source('backtest.R')


# Subset by decade and check how coefficients compare for model m3
# Try to have GARCH model capture seasonality in the residuals
# Ljung box will always reject independence on garch residuals
# Try garch extension for leverage effects.
# Use garchfit to have it all in one model.
# Test normality of residuals

chevron <- read.csv('CVX.csv')
head(chevron)

chevron.adj.close.ts <- ts(chevron$Adj.Close,
              start = c(1962, 1),
              frequency = 252)

# Log transformation is required to make it additive
autoplot(chevron.adj.close.ts)

chevron.log <- log(chevron.adj.close.ts)

# Trend stationary or random walk with a drift?
autoplot(chevron.log)

# Checking length of series
length(chevron.log)

# Splitting into train and test datasets
chevron.train <- subset(chevron.log, end = 13900)
chevron.test <- subset(chevron.log, start = 13901)

# Decomposing
autoplot(decompose(chevron.train))

# These look like random walk, but that might be just because of the trend
acf(chevron.train)
pacf(chevron.train)

# Unit root
eacf(chevron.train)

# Dickey fuller barely rejects the null hypothesis of unit root with a trend
# (both at lag 10 and 15), but kpss rejects the null hypothesis of stationarity 
# with a trend. This could be a random walk with drift, and I will start 
# exploring it as such.
adfTest(chevron.train, lags = 10, type = 'ct')
adfTest(chevron.train, lags = 15, type = 'ct')
kpss.test(chevron.train, null = 'Trend')

# Now it definitely looks stationary
autoplot(diff(chevron.train))

# T test rejects the null hypothesis, meaning the sample mean and the mean of the
# population differ, indicating a drift.
t.test(diff(chevron.log))

# Dickey fuller  now definitely rejects the null hypothesis of unit root and
# KPSS fails to reject the null hypothesis of stationarity.
adfTest(diff(chevron.train), lags = 10, type = 'nc')
adfTest(diff(chevron.train), lags = 15, type = 'nc')
kpss.test(diff(chevron.train), null = 'Level')

# No apparent AR behavior and some auto correlation after lag 0.
acf(diff(chevron.train))

# Looks like a high order MA considering that there does not seem to be seasonality.
# Looking at it up close it is much clearer that there is auto correlation after lag 0.
pacf(diff(chevron.train))

# This is unclear. Maybe an MA(4)?
eacf(diff(chevron.train))

# Trying a first model
m0 <- Arima(chevron.train, order = c(0,1,4), include.drift = TRUE)

# MA 4 and drift are significant, as expected, and MA2, surprisingly, is also
# significant
coeftest(m0)

# Removing insignificant terms from first model
m0 <- Arima(chevron.train, order = c(0,1,4), include.drift = TRUE,
            fixed = c(0, NA, 0, NA, NA))

coeftest(m0)

# Visually it seems ok.
fc1 <- forecast(m0, h = length(chevron.test))
plot(fc1)
lines(chevron.test, col = 'red')

# RMSFE of .22
rmsfe.m0 <- sqrt(mean((fc1$mean - chevron.test)^2))
rmsfe.m0

# About 3.5%, not bad.
mape.m0 <- mean(abs((fc1$mean - chevron.test)/chevron.test))
mape.m0

# 
bt0 <- backtest(m0, chevron.log, orig = .8* length(chevron.log), h = 1)


# Residuals are definitely stationary, but there's some volatility.
autoplot(m0$residuals)

# Looks very close to white noise, except for a few instances of auto correlation
acf(m0$residuals)

# Looking at it closer there is auto correlation at four different lags
pacf(m0$residuals)

# L-Jung box test barely fails to reject the null hypothesis of independence
Box.test(m0$residuals, lag = 10, type = 'Ljung')

# Square residuals show a lot of volatility
autoplot(m0$residuals^2)
acf(m0$residuals^2)

res <- m0$residuals

# Trying a garch model instead as, judging by the pacf of the squared residuals
# it would take a really high order arch model to get rid of auto correlation.
# The coefficients are all very significant
garch.fit <- garch(res, order = c(1,1))
garch.fit
coeftest(garch.fit)

# These residuals are looking better but there is still some auto correlation
autoplot(garch.fit$residuals^2)
Acf(garch.fit$residuals^2)

# Ljung box test definitely rejects the null hypothesis
Box.test(garch.fit$residuals, type = 'Ljung')

# If garch didn't manage to make the residuals white noise, let's increase the
# order of the arima/try different models.

# =============================================================================

# Checking on what auto.arima comes up with
m1 <- auto.arima(chevron.train)

# AR(2) and drift are very significant, AR(1) not so much.
coeftest(m1)

# Keeping only significant terms
m1 <- Arima(chevron.train, order = c(2, 1, 0), include.drift = TRUE,
            fixed = c(0, NA, NA))

coeftest(m1)

# Visually it seems ok.
fc2 <- forecast(m1, h = length(chevron.test))
plot(fc2)
lines(chevron.test, col = 'red')

# RMSFE of .22, same as m0
rmsfe.m1 <- sqrt(mean((fc2$mean - chevron.test)^2))
rmsfe.m1

# About 3.5%, same as m0.
mape.m1 <- mean(abs((fc2$mean - chevron.test)/chevron.test))
mape.m1

# "Error in if (orig > T) orig = T : the condition has length > 1"
bt1 <- backtest(m1, chevron.log, orig = .8* length(chevron.log), h = 1)

# Residuals are definitely stationary, but there's some volatility.
autoplot(m1$residuals)

# Looks very close to white noise, except for a few instances of auto correlation
acf(m1$residuals)

# Looking at it closer there is auto correlation at four different lags
pacf(m1$residuals)

# L-Jung box test rejects the null hypothesis of independence
Box.test(m1$residuals, lag = 10, type = 'Ljung')

# Square residuals show a lot of volatility
autoplot(m1$residuals^2)
acf(m1$residuals^2)

res <- m1$residuals

# Trying a garch model instead as, judging by the pacf of the squared residuals
# it would take a really high order arch model to get rid of auto correlation.
# The coefficients are all very significant
garch.fit <- garch(res, order = c(1,1))
garch.fit
coeftest(garch.fit)

# These residuals are looking better but there is still some auto correlation
autoplot(garch.fit$residuals^2)
Acf(garch.fit$residuals^2)

# Ljung box test definitely rejects the null hypothesis
Box.test(garch.fit$residuals, type = 'Ljung')

# =============================================================================

# Checking auto.arima using bic.
m2 <- auto.arima(chevron.train, ic = 'bic')
m2

# It seems to not return any coefficients!
coeftest(m2)

# =============================================================================
# This is the best model. RMSFE and MAPE are about the same as all other ones,
# but residuals are much better behaved. Not to mention this one is more parsimonious
# than many other ones I tried.
# Trying a higher order model

# SUBSET BY DECADE AND CHECK ON COEFFICIENTS.
m3 <- Arima(chevron.train, order = c(2,1,2), include.drift = TRUE)

# All very significant!
coeftest(m3)

# Visually it seems ok.
fc3 <- forecast(m3, h = length(chevron.test))
plot(fc3)
lines(chevron.test, col = 'red')

# RMSFE of .22
rmsfe.m3 <- sqrt(mean((fc3$mean - chevron.test)^2))
rmsfe.m3

# About 3.5%, not bad.
mape.m3 <- mean(abs((fc3$mean - chevron.test)/chevron.test))
mape.m3

# "Error in solve.default(res$hessian * n.used, A) :
# Lapack routine dgesv: system is exactly singular: U[1,1] = 0"
bt3 <- backtest(m3, chevron.log, orig = .8* length(chevron.log), h = 1)

# Residuals are definitely stationary, but there's some volatility.
autoplot(m3$residuals)

# Looks very close to white noise, except for a few instances of auto correlation
acf(m3$residuals)

# Residuals seem better than the ones for the previous models!
pacf(m3$residuals)

# L-Jung box test fails to reject the null hypothesis of independence by quite
# a lot! Finally!
Box.test(m3$residuals, lag = 10, type = 'Ljung')

# TRY TO HAVE GARCH MODEL SEASONALITY
# Square residuals show a lot of volatility
autoplot(m3$residuals^2)
acf(m3$residuals^2)

res <- m3$residuals

# Trying a garch model instead as, judging by the pacf of the squared residuals
# it would take a really high order arch model to get rid of auto correlation.
# The coefficients are all very significant
garch.fit <- garch(res, order = c(1,1))
garch.fit
coeftest(garch.fit)

# These residuals are looking better but there is still some auto correlation
autoplot(garch.fit$residuals^2)
Acf(garch.fit$residuals^2)

# This is fine! Try extension for leverage effects.
# Use garchfit to have it all in one model.
# Ljung box test still definitely rejects the null hypothesis, but again, closer
# to failing to reject than other models
Box.test(garch.fit$residuals^2, type = 'Ljung')

# Evaluate normality of residuals here
res <- na.omit(garch.fit$residuals)
ggplot(res, aes(x=res)) + geom_histogram()

ggplot(res, aes(sample=res)) +
  stat_qq() + 
  stat_qq_line()

jarque.bera.test(na.omit(res)) # rejects normality

# GARCH modelling
# garchfit produces weird results, stick with ugarch
gfit3 = garchFit( ~ arma(2, 2) + garch(1, 1), data = diff(chevron.train), trace = F)
gfit3 # AR2 and MA2 are insignificant now.

resgfit3 <- gfit3@residuals/gfit3@sigma.t
autoplot(ts(resgfit3^2))
Acf(resgfit3^2)
Box.test(resgfit3^2, lag = 15, type = 'Ljung')

# This looks better
s = ugarchspec(variance.model=list(garchOrder=c(1, 1)),
               mean.model=list(armaOrder=c(2, 2)))

gfit4 <- ugarchfit(s, diff(chevron.log))

# Everything is significant. Robust AR2, MA2, omega and alpha are not significant.
gfit4@fit
gfit4res <- residuals(gfit4, standardize=T)
autoplot(gfit4res^2)
acf(gfit4res^2)

# Almost fails to reject independence at a 95% confidence interval! .04
Box.test(gfit4res^2, lag = 15, type = 'Ljung')

ggplot(gfit4res, aes(x=gfit4res)) + geom_histogram()

ggplot(gfit4res, aes(sample=gfit4res)) +
  stat_qq() + 
  stat_qq_line()

skewness(gfit4res) # Very small skewness, -0.15
kurtosis(gfit4res) # Not bad, 2.19

jarque.bera.test(na.omit(gfit4res)) # rejects normality

gfit4.rolltest <- ugarchroll(s, data = diff(chevron.log), n.start = 2000,
                                            refit.window = 'moving',
                                            refit.every = 500)

report(gfit4.rolltest, type = 'fpm')

plot(ugarchforecast(gfit4, n.ahead=10))


# =============================================================================

# Trying a lower order model
m4 <- Arima(chevron.train, order = c(0,1,2), include.drift = TRUE)

# MA(2) and drift are significant
coeftest(m4)

# Visually it seems ok.
fc4 <- forecast(m4, h = length(chevron.test))
plot(fc4)
lines(chevron.test, col = 'red')

# RMSFE of .22
rmsfe.m4 <- sqrt(mean((fc4$mean - chevron.test)^2))
rmsfe.m4

# About 3.5%, not bad.
mape.m4 <- mean(abs((fc4$mean - chevron.test)/chevron.test))
mape.m4

# "Error in if (orig > T) orig = T : the condition has length > 1"
bt4 <- backtest(m4, chevron.log, orig = .8* length(chevron.log), h = 1)

# Residuals are definitely stationary, but there's some volatility.
autoplot(m4$residuals)

# Looks very close to white noise, except for a few instances of auto correlation
acf(m4$residuals)

# hmm
pacf(m4$residuals)

# L-Jung box test rejects the null hypothesis of independence
Box.test(m4$residuals, lag = 10, type = 'Ljung')

# Square residuals show a lot of volatility
autoplot(m4$residuals^2)
acf(m4$residuals^2)

res <- m4$residuals

# Trying a garch model instead as, judging by the pacf of the squared residuals
# it would take a really high order arch model to get rid of auto correlation.
# The coefficients are all very significant
garch.fit <- garch(res, order = c(1,1))
garch.fit
coeftest(garch.fit)

# These residuals are looking better but there is still some auto correlation
autoplot(garch.fit$residuals^2)
Acf(garch.fit$residuals^2)

# Ljung box test still definitely rejects the null hypothesis
Box.test(garch.fit$residuals, type = 'Ljung')

# =============================================================================


# Trying a lower order model
m5 <- Arima(chevron.train, order = c(2,1,6), include.drift = TRUE)

# AR(2), MA(2), MA(6) and drift are significant
coeftest(m5)

# Removing insignificant terms from first model
m5 <- Arima(chevron.train, order = c(2,1,6), include.drift = TRUE,
            fixed = c(0, NA, 0, NA, 0, 0, 0, NA, NA))

# Visually it seems ok.
fc5 <- forecast(m5, h = length(chevron.test))
plot(fc5)
lines(chevron.test, col = 'red')

# RMSFE of .22
rmsfe.m5 <- sqrt(mean((fc5$mean - chevron.test)^2))
rmsfe.m5

# About 3.5%, not bad.
mape.m5 <- mean(abs((fc5$mean - chevron.test)/chevron.test))
mape.m5

# "Error in if (orig > T) orig = T : the condition has length > 1"
bt5 <- backtest(m5, chevron.log, orig = .8* length(chevron.log), h = 1)

# Residuals are definitely stationary, but there's some volatility.
autoplot(m5$residuals)

# Looks very close to white noise, except for a few instances of auto correlation
acf(m5$residuals)

# Looks a little better than most models
pacf(m5$residuals)

# L-Jung box test fails to rejects the null hypothesis of independence, but barely
Box.test(m5$residuals, lag = 10, type = 'Ljung')

# Square residuals show a lot of volatility
autoplot(m5$residuals^2)
acf(m5$residuals^2)

res <- m5$residuals

# Trying a garch model instead as, judging by the pacf of the squared residuals
# it would take a really high order arch model to get rid of auto correlation.
# The coefficients are all very significant
garch.fit <- garch(res, order = c(1,1))
garch.fit
coeftest(garch.fit)

# These residuals are looking better but there is still some auto correlation
autoplot(garch.fit$residuals^2)
Acf(garch.fit$residuals^2)

# Ljung box test still definitely rejects the null hypothesis
Box.test(garch.fit$residuals, type = 'Ljung')

# =============================================================================

# As nothing seems to work with an ARIMA model, I'll try an ARMA model instead.
# Maybe it is trend stationary after all.
m6 <- Arima(chevron.train, order = c(0,0,4), include.drift = TRUE)

# All very significant
coeftest(m6)

# Visually not as appropriate as all previous models.
fc6 <- forecast(m6, h = length(chevron.test))
plot(fc6)
lines(chevron.test, col = 'red')

# RMSFE of .35
rmsfe.m6 <- sqrt(mean((fc6$mean - chevron.test)^2))
rmsfe.m6

# About 6.4%, almost twice the error of all other models.
mape.m6 <- mean(abs((fc6$mean - chevron.test)/chevron.test))
mape.m6

# "Error in if (orig > T) orig = T : the condition has length > 1"
bt6 <- backtest(m6, chevron.log, orig = .8* length(chevron.log), h = 1)

# Residuals are definitely NOT stationary. I'll stop here.
autoplot(m6$residuals)