import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 5))
ax.axis('off')

layers = [
    'Input: 9x9x30 patch',
    'Conv2D(64, 3x3) + ReLU',
    'Conv2D(128, 3x3) + ReLU',
    'AdaptiveAvgPool2D(1x1)',
    'Flatten',
    'Dense(128) + ReLU + Dropout(0.5)',
    'Dense(num_classes) + Softmax'
]

x = [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15]
y = [0.70, 0.54, 0.38, 0.22, 0.06, -0.10, -0.26]

for idx, layer in enumerate(layers):
    rect = plt.Rectangle((0.06, y[idx]), 0.5, 0.12, fill=True, color='skyblue', edgecolor='k')
    ax.add_patch(rect)
    ax.text(0.09, y[idx] + 0.06, layer, fontsize=11, verticalalignment='center')

# arrows
for i in range(len(layers)-1):
    ax.annotate('', xy=(0.31, y[i]-0.02), xytext=(0.31, y[i+1]+0.12), arrowprops=dict(arrowstyle='->', lw=2))

ax.set_xlim(0, 1)
ax.set_ylim(-0.3, 1)
ax.set_title('SimpleCNN Architecture (2D-CNN with PCA input)', fontsize=14, pad=20)

plt.savefig('architecture.png', dpi=200, bbox_inches='tight')
plt.close()
print('Saved architecture.png')