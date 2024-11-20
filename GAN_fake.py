from keras.layers import InputLayer,Sequential,Dense,Dropout 
import numpy as np 
import pandas as pd 

np.random.seed(101)


# Define the generator network
def build_generator(latent_dim, output_dim):
    model = Sequential()
    model.add(Dense(64, input_shape=(latent_dim,)))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(8,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim, activation='sigmoid'))
    return model

# Define the discriminator network
def build_discriminator(input_dim):
    model = Sequential()
    model.add(InputLayer(input_shape = (input_dim,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    return model

# Dimensionality of the input noise for the generator
latent_dim = 128
generator = build_generator(latent_dim, fraud_data.shape[1])
discriminator = build_discriminator(fraud_data.shape[1])

print(generator.summary())
print(discriminator.summary())


import keras.backend as K
def precision(y_true,y_pred):
    true_positives = K.sum(K.round(K.clip(y_true*y_pred,0,1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred,0,1)))
    precision_value = true_positives/(predicted_positives+K.epsilon())
    return precision_value

def recall(y_true,y_pred):
    true_positives = K.sum(K.round(K.clip(y_true*y_pred,0,1)))
    possible_positives = K.sum(K.round(K.clip(y_true,0,1)))
    recall_value = true_positives/(possible_positives+K.epsilon())
    return recall_value

discriminator.compile(
    optimizer=Adam(lr=0.0001,beta_1=0.5),
    loss='binary_crossentropy',  
    metrics=[precision,recall]
)
def generator_loss_log_d(y_true, y_pred):
    return - K.mean(K.log(y_pred + K.epsilon()))

# Build and compile the GANs upper optimization loop combining generator and discriminator
def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(optimizer=Adam(lr=0.0001, beta_1=0.5),
    loss=generator_loss_log_d)
    return model

# Call the upper loop function
gan = build_gan(generator, discriminator)

print(gan.summary())
# Set hyperparameters
epochs = 10000
batch_size = 8

# Training loop for the GANs
for epoch in range(epochs):
    # Train discriminator (freeze generator)
    discriminator.trainable = True
    generator.trainable = False

    # Random sampling from the real fraud data
    real_fraud_samples = fraud_data[np.random.randint(0, num_real_fraud1, batch_size)]

    # Generate fake fraud samples using the generator
    noise = np.random.normal(0, 1, size=(batch_size, latent_dim))
    fake_fraud_samples = generator.predict(noise)

    # Create labels for real and fake fraud samples
    real_labels = np.ones((batch_size, 1))
    fake_labels = np.zeros((batch_size, 1))

    # Train the discriminator on real and fake fraud samples
    d_loss_real = discriminator.train_on_batch(real_fraud_samples, real_labels)
    d_loss_fake = discriminator.train_on_batch(fake_fraud_samples, fake_labels)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # Train generator (freeze discriminator)
    discriminator.trainable = False
    generator.trainable = True

    # Generate synthetic fraud samples and create labels for training the generator
    noise = np.random.normal(0, 1, size=(batch_size, latent_dim))
    valid_labels = np.ones((batch_size, 1))

    # Train the generator to generate samples that "fool" the discriminator
    g_loss = gan.train_on_batch(noise, valid_labels)

    # Print the progress
    if epoch % 100 == 0:
        print("Epoch: {} - D Loss: {} - G Loss: {}".format(epoch,d_loss,g_loss))


noise = np.random.normal(0,1,size = (int(num_synthetic_samples1),int(latent_dim)))

synthetic_fraud_data = generator.predict(noise)
fake_df_cbs = pd.DataFrame(synthetic_fraud_data,columns = features)
fake_df_cbs.shape