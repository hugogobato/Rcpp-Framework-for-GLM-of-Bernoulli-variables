library(Rcpp)
library(RcppEigen)
library(stats)
sourceCpp("List_3_Ex_10_Rcpp.cpp")
# Set up a 1x2 plot layout
par(mfrow=c(1,1))

# Hyperparameters
num_obs <- 10000         # Number of observations
num_x <- 10             # Number of true predictors in the DGP
num_x_false <- 1       # Number of unrelated predictors (false predictors)

# Generate the true predictors (x) as part of the DGP
x <- matrix(rnorm(num_obs * num_x), nrow = num_obs, ncol = num_x)
# Generate the intercept column
intercept <- rep(1, num_obs)
# Combine intercept and true predictors
x <- cbind(intercept, x)
# Generate coefficients for x (true predictors)
beta <- runif(num_x+1, min = -1, max = 1)  # Random coefficients between -1 and 1

# Create a data frame for beta coefficients
beta_names <- c("intercept", paste0("x", 1:num_x))
beta_df <- data.frame(Predictor = beta_names, Coefficient = beta)

# Generate the linear predictor and apply the logistic function
linear_pred <- x %*% beta
prob <- 1 / (1 + exp(-linear_pred))  # Logistic function

# Generate y from a Bernoulli distribution based on the probabilities
y <- rbinom(num_obs, 1, prob)

# Generate unrelated predictors (x_false)
x_false <- matrix(rnorm(num_obs * num_x_false), nrow = num_obs, ncol = num_x_false)

# Combine data into a single data frame for easy handling
data <- data.frame(y = y, x, x_false)
names(data)[2:(num_x + 2)] <- c("intercept", paste0("x", 1:(num_x)))  # Rename true predictors
names(data)[(num_x + 3):(num_x + num_x_false + 2)] <- paste0("x_false", 1:num_x_false)  # Rename false predictors

print(beta_df)

number_iterations <- 10
x_original <- as.matrix(data[,2:ncol(data)])
x_test <- as.matrix(data[,2:ncol(data)-1])
y <- as.numeric(data$y)
start_time <- Sys.time()
beta_vector <- beta_estimation(y,x_original,number_iterations)
cat("\n")
cat("Estimated Beta parameters:", beta_vector, "\n")
cat("\n")
end_time <- Sys.time()
time_taken_beta_my_code <- end_time - start_time

x_seq <- create_x_sequence(x_original, 100)

#print(head(x_seq))

#y_pred <- predicted_y(x_seq, beta_vector)

# Plot the original data
#plot(x_original[,2], y, main = "Bernoulli GLM with Canonical Link", xlab = "x", ylab = "y")
#lines(x_seq[,2], y_pred, col = "blue", lwd = 2)

start_time <- Sys.time()
Q_LR <- Q_LR(y,x_original,x_test,number_iterations)
Q_LR_p_value <- pchisq(Q_LR, df = 1, lower.tail = FALSE)
end_time <- Sys.time()
time_taken_LR_my_code <- end_time - start_time



Q_SR <- Q_SR(y,x_test,number_iterations)
Q_SR_p_value <- pchisq(Q_SR, df = 1, lower.tail = FALSE)
Q_W <- Q_W(y,x_original,ncol(x_original),number_iterations)
Q_W_p_value <- pchisq(Q_W, df = 1, lower.tail = FALSE)
Q_G <- Q_G(y,x,x_test,number_iterations)
Q_G_p_value <- pchisq(Q_G, df = 1, lower.tail = FALSE)

cat("Estimated Statistic and p-value for Q_LR:", Q_LR, "and", Q_LR_p_value, "\n")
cat("Estimated Statistic and p-value for Q_SR:", Q_SR, "and", Q_SR_p_value, "\n")
cat("Estimated Statistic and p-value for Q_W:", Q_W, "and", Q_W_p_value, "\n")
cat("Estimated Statistic and p-value for Q_G:", Q_G, "and", Q_G_p_value, "\n")
cat("\n")
start_time <- Sys.time()
# Full model (with beta1)
model_full <- glm(y ~ x_original, family = binomial(link = "logit"),control = glm.control(maxit =number_iterations))
# Print only the estimated beta coefficients
beta_estimates <- summary(model_full)$coefficients[, 1]  # Extract the first column of coefficients
print(beta_estimates)
cat("\n")
end_time <- Sys.time()
time_taken_beta_glm_package <- end_time - start_time

start_time <- Sys.time()
# Null model (without beta1, i.e., only intercept)
model_null <- glm(y ~ x_test, family = binomial(link = "logit"),control = glm.control(maxit =number_iterations))

# Likelihood Ratio Test
lrt_statistic <- 2 * (logLik(model_full) - logLik(model_null))
lrt_p_value <- pchisq(lrt_statistic, df = 1, lower.tail = FALSE)
cat("Likelihood Ratio Test Statistic:", lrt_statistic, "\n")
cat("Likelihood Ratio Test p-value:", lrt_p_value, "\n")
end_time <- Sys.time()
time_taken_LR_glm_package <- end_time - start_time


# Wald Test
beta_1 <- coef(summary(model_full))["x_originalx_false1", "Estimate"]
se_beta_1 <- coef(summary(model_full))["x_originalx_false1", "Std. Error"]
wald_statistic <- (beta_1 / se_beta_1) ^ 2
wald_p_value <- pchisq(wald_statistic, df = 1, lower.tail = FALSE)
cat("Wald Test Statistic:", wald_statistic, "\n")
cat("Wald Test p-value:", wald_p_value, "\n")

time_taken_beta_glm_package <- as.numeric(time_taken_beta_glm_package, units = "secs")
time_taken_beta_my_code <- as.numeric(time_taken_beta_my_code, units = "secs")
time_taken_LR_glm_package <- as.numeric(time_taken_LR_glm_package, units = "secs")
time_taken_LR_my_code <- as.numeric(time_taken_LR_my_code, units = "secs")

cat("How faster is my code compared to GLM package for Beta estimation:", time_taken_beta_glm_package/time_taken_beta_my_code, "\n")
cat("How faster is my code compared to GLM package for performing the Likelihood Ratio test:", time_taken_LR_glm_package/time_taken_LR_my_code, "\n")
