# library(usethis)
# edit_r_environ()
#  use_github(protocol = "https",
#             auth_token = Sys.getenv("GITHUB_PAT"))



    library(dplyr)
    library(stringr)
    library(scales)
    
# load accounting fraud data     
    dataset <- read.csv("uscecchini28.csv")
    data_dictionary <- read.csv("get_Data_dict.csv") #data dictionary: http://www.crsp.org/products/documentation/annual-data-industrial
    

#  change column names of "dataset", to reflect names of accounting variables
        colNames_dataset <- toupper(names(dataset))
        var_in_dict <-    data.frame("var"=toupper(colNames_dataset)) %>% 
                          left_join(data_dictionary, by=c("var"="New.CCM.Item.Name")) %>% 
                          filter(!is.na(Description)) %>% 
                          select(var,Description)  
        
        var_position <- c()
        for (i in var_in_dict$var){
          var_position <- c(var_position,which(colNames_dataset==i))
        }
        var_in_dict$position <- var_position
    
        
        for (i in 1:dim(var_in_dict)[1]){
          names(dataset)[var_in_dict[i,3]] <- var_in_dict[i,2]
        }
        names(dataset) <- gsub("\\\\","",names(dataset))

        
# some data summary
    
    str(dataset) #why is this dataset ignoring so many other variables? any justificaitons?
    dim(dataset)
    
    #some numbers about frauds
        fraud_companies <- unique((dataset[dataset$understatement==1,]$gvkey))
        cat("num of fraud companies: ", length(fraud_companies))
        prop.table(table(dataset$understatement)) #fraud reporting (understatement=1) /all reporting
        length(fraud_companies)/length(unique(dataset$gvkey)) #num. fraud companies/ all companies
        round(prop.table(table(dataset$issue)),2) #what is this? what "issue" are we talking about?
    
    #check missing values
        missing_value_count <- data.frame(missing_value_count= apply(dataset, 2, function(x){sum(is.na(x))})) %>% 
                                filter(missing_value_count>=1)
        missing_value_count$missing_var <- row.names(missing_value_count)
        row.names(missing_value_count) <- NULL
        missing_value_count <- missing_value_count[,c(2,1)]
        missing_vale_df <- missing_value_count %>% 
                              arrange(missing_value_count) %>%
                              mutate(missing_value_perc= percent(missing_value_count/nrow(dataset)))
 
        #check how often fraud companies have missing values
        
        
        
        
    
    #some visualizaiton 
        library(ggplot2)
    
        ggplot(y, aes(x = start_station_name, y = duration, main="Car Distribution"),data=dataset) +
          geom_bar(stat = "identity") +
          coord_flip() + scale_y_continuous(name="Average Trip Duration (in seconds)") +
          scale_x_discrete(name="Start Station") +
          theme(axis.text.x = element_text(face="bold", color="#008000",
                                           size=8, angle=0),
                axis.text.y = element_text(face="bold", color="#008000",
                                           size=8, angle=0))
    #split into training and test sets
        train_data <- dataset$train$x 
        train_targets <- dataset$train$y 
        test_data <- dataset$test$x 
        test_targets <- dataset$test$y
        
        mean <- apply(train_data,2,mean)
        std <- apply(train_data,2, sd)    
        
        train_data <- scale(train_data, center=mean,scale=std)
        test_data <- scale(test_data, center=mean,scale=std) 
    
# where is the data dicitonary...
# does the performance metrics make sense.. 
    
# quesitons: how do we deal with varying age of the businesses, varying time since IPO,
    # can we distinguish different types of accounting fraud (e.g. overstatement of certain lines)
    # numbers about current asset vs. liabilities, distribution of assets? 
    # what types of account fraud have the companies committed? How do they reflect on the books? are those relevant variables available in the data?

# vae encoder network (ref: Deep Learning with R, by Francois Chollet and Joseph J. Allaire)
    #we want to generate fraud cases
      # some features we might be interested in: peer-group performance, short-term liability, typical accounting fraud variables

    library(keras)
    library(tensorflow)
    use_condaenv("r-reticulate", required = TRUE) #https://community.rstudio.com/t/error-installation-of-tensorflow-not-found-in-rstudio/67200/2
    reticulate::py_config()
    
    K.clear_session()
    tf.compat.v1.reset_default_graph()
    K <- backend()
    img_shape <- c(28, 28, 1)
    batch_size <- 16
    latent_dim <- 2L
    input_img <- layer_input(shape = img_shape)
    x <- input_img %>%
      layer_conv_2d(filters = 32, kernel_size = 3, padding = "same",
                    activation = "relu") %>%
      layer_conv_2d(filters = 64, kernel_size = 3, padding = "same",
                    activation = "relu", strides = c(2, 2)) %>%
      layer_conv_2d(filters = 64, kernel_size = 3, padding = "same",
                    activation = "relu") %>%
      layer_conv_2d(filters = 64, kernel_size = 3, padding = "same",
                    activation = "relu")
    shape_before_flattening <- K$int_shape(x)
    x <- x %>%
      layer_flatten() %>%
      layer_dense(units = 32, activation = "relu")
    z_mean <- x %>%
      layer_dense(units = latent_dim)
    z_log_var <- x %>%
      layer_dense(units = latent_dim)
    #latent space sampling function 
    sampling <- function(args) {
      c(z_mean, z_log_var) %<-% args
      epsilon <- K$random_normal(shape = list(K$shape(z_mean)[1], latent_dim),
                                 mean = 0, stddev = 1)
      z_mean + K$exp(z_log_var) * epsilon
    }
    z <- list(z_mean, z_log_var) %>%
      layer_lambda(sampling)
    
    # VAE decoder network, mapping latent space points to images
    decoder_input <- layer_input(K$int_shape(z)[-1])
    x <- decoder_input %>%
      layer_dense(units = prod(as.integer(shape_before_flattening[-1])),
                  activation = "relu") %>%
      layer_reshape(target_shape = shape_before_flattening[-1]) %>%
      layer_conv_2d_transpose(filters = 32, kernel_size = 3, padding = "same",
                              activation = "relu", strides = c(2, 2)) %>%
      layer_conv_2d(filters = 1, kernel_size = 3, padding = "same",
                    activation = "sigmoid")
    decoder <- keras_model(decoder_input, x)
    z_decoded <- decoder(z)
    
    #custome layer to compute VAE
    library(R6)
    CustomVariationalLayer <- R6Class("CustomVariationalLayer",
                                      inherit = KerasLayer,
                                      public = list(
                                        vae_loss = function(x, z_decoded) {
                                          x <- K$flatten(x)
                                          z_decoded <- K$flatten(z_decoded)
                                          xent_loss <- metric_binary_crossentropy(x, z_decoded)
                                          kl_loss <- -5e-4 * K$mean(
                                            1 + z_log_var - K$square(z_mean) - K$exp(z_log_var),
                                            axis = -1L
                                          )
                                          K$mean(xent_loss + kl_loss)
                                        },
                                        call = function(inputs, mask = NULL) {
                                          x <- inputs[[1]]
                                          z_decoded <- inputs[[2]]
                                          loss <- self$vae_loss(x, z_decoded)
                                          self$add_loss(loss, inputs = inputs)
                                          x
                                        }
                                      )
    )
    layer_variational <- function(object) {
      create_layer(CustomVariationalLayer, object, list())
    }
    y <- list(input_img, z_decoded) %>%
      layer_variational()
    #training VAE
    vae <- keras_model(input_img, y)
    vae %>% compile(
      optimizer = "rmsprop",
      loss = NULL,experimental_run_tf_function=FALSE
    )
    mnist <- dataset_mnist()
    c(c(x_train, y_train), c(x_test, y_test)) %<-% mnist
    x_train <- x_train / 255
    x_train <- array_reshape(x_train, dim =c(dim(x_train), 1))
    x_test <- x_test / 255
    x_test <- array_reshape(x_test, dim =c(dim(x_test), 1))
    
    
    vae %>% fit(
      x = x_train, y = NULL,
      epochs = 10,
      batch_size = batch_size,
      validation_data = list(x_test, NULL)
    ) #https://github.com/rstudio/keras/issues/1008
    
    #Sampling a grid of points from the 2D latent space and decoding them to images
    n <- 15
    digit_size <- 28
    grid_x <- qnorm(seq(0.05, 0.95, length.out = n))
    grid_y <- qnorm(seq(0.05, 0.95, length.out = n))
    op <- par(mfrow = c(n, n), mar = c(0,0,0,0), bg = "black")
    for (i in 1:length(grid_x)) {
      yi <- grid_x[[i]]
      for (j in 1:length(grid_y)) {
            xi <- grid_y[[j]]
            z_sample <- matrix(c(xi, yi), nrow = 1, ncol = 2)
            z_sample <- t(replicate(batch_size, z_sample, simplify = "matrix"))
            x_decoded <- decoder %>% predict(z_sample, batch_size = batch_size)
            digit <- array_reshape(x_decoded[1,,,], dim = c(digit_size, digit_size))
            plot(as.raster(digit))
      }
    }
    par(op)
    
    
    
   #