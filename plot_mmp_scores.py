import matplotlib.pyplot as plt

fig = plt.figure(figsize=(7, 3.5), dpi=100)

x_labels = [5, 10, 25, 50, 100, 200, 400, 800, 1600]
y1 = [-0.031, -0.002, 0.4450, 0.613, 0.606, 0.673, 0.679, 0.622, 0.668]
y13 = [0.002, 0.246, 0.220, 0.612, 0.636, 0.577, 0.635, 0.652, 0.645]
y14 = [0.396, 0.113, 0.491, 0.488, 0.510, 0.552, 0.572, 0.571, 0.558]
y16 = [0.300, 0.402, 0.331, 0.569, 0.593, 0.558, 0.611, 0.611, 0.602]
y24 = [-0.070, 0.424, 0.160, 0.457, 0.487, 0.584, 0.576, 0.584, 0.573]

plt.plot(x_labels, y1, label="MMP1")
plt.plot(x_labels, y13, label="MMP13")
plt.plot(x_labels, y14, label="MMP14")
plt.plot(x_labels, y16, label="MMP16")
plt.plot(x_labels, y24, label="MMP24")
plt.legend()

for i in range(6, 9):
    plt.text(x_labels[i], y1[i], y1[i], ha="center", va="bottom", fontsize=10)
    plt.text(x_labels[i], y13[i], y13[i], ha="center", va="bottom", fontsize=10)
    plt.text(x_labels[i], y14[i], y14[i], ha="center", va="bottom", fontsize=10)
    plt.text(x_labels[i], y16[i], y16[i], ha="center", va="bottom", fontsize=10)
    plt.text(x_labels[i], y24[i], y24[i], ha="center", va="bottom", fontsize=10)

plt.xlabel("Hidden Layer Size")
plt.ylabel("R^2 Score")
plt.title("MMP R^2 Scores")
plt.show()
