This dissertation submitted to JNTUH, Hyderabad in partial fulfillment of the academic requirements for the award of the degree:
**Bachelor of Technology**
in
**Computer Science and Engineering (AI&ML)**

Also presented at
**5th International Conference on Recent Trends in Engineering, Technology and Management 2025 - ICRETM'25**

Contibutors
1. Mohammed Tauseef Ahmed
2. Eppa Srujan Reddy
3. Shaik Shoaib Hannan

Refer official documentation for detailed report

# LoMar - A Local Defence Against Poisoning Attack on Federated Learning

Federated Learning (FL) offers an efficient decentralized machine learning framework where the training data remains distributed across remote clients in a network, ensuring privacy and security, particularly for mobile edge computing using IoT devices. However, recent studies have highlighted that this decentralized approach is vulnerable to poisoning attacks from malicious clients, which can compromise the integrity of the model. To counter these attacks, we propose a two-phase defense algorithm known as Local Malicious Factor (LoMar). In the first phase, LoMar evaluates model updates from each remote client by measuring the relative distribution of their updates compared to their neighboring clients. This is done using a kernel density estimation method, which helps to detect outlier updates that are likely to be malicious. In the second phase, LoMar approximates an optimal threshold to statistically distinguish between malicious and clean updates, ensuring that harmful contributions are filtered out while preserving the integrity of the model. To assess the effectiveness of our defense strategy, we conducted comprehensive experiments on four real-world datasets. The results show that LoMar can significantly enhance the defense capabilities of FL systems against poisoning attacks. Notably, in experiments on the Amazon dataset under a label-flipping attack, LoMar increased the target label testing accuracy from 96.0% to 98.8% and the overall average testing accuracy from 90.1% to 97.0%, outperforming existing defense methods such as FG+Krum. These results demonstrate that LoMar is a highly effective defense mechanism for protecting FL systems.
