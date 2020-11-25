import pickle
import matplotlib.pyplot as plt

loss = []
with open('train_loss','rb') as f:
    loss = pickle.load(f)

plt.figure()
plt.xlabel("Number of Iterations / 10")
plt.ylabel("Training loss")
plt.plot(loss)
plt.savefig("Training_Loss.jpg")
