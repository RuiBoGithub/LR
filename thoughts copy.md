`Transfer learning`

Dyna-PINN: Physics-informed deep dyna-q reinforcement learning for intelligent control of building heating system in low-diversity training data regimes

Transfer learning with deep neural networks for model predictive control of HVAC and natural ventilation in smart buildings

----
Our study highlights the substantial benefits of incorporating physics-informed models within the DDQ framework, particularly in **low-diversity data** regimes. The DDQ approaches consistently outperformed the model-free DQN in low-diversity data regimes, resulting in lower thermal discomfort and energy use. This was primarily due to the two-fold training approach of the DDQ controllers, where learning from real experiences was coupled with simulated experiences, thus alleviating the issue of low-diversity training data availability.

----
Both physics-based control and traditional reinforcement learning (RL) methods often face challenges in low-diversity data scenarios, which are common in building control applications. These challenges arise due to the high cost of real-world experimentation and the potential disruption to occupants' thermal comfort [32]. A known approach for addressing this in MPC-based heating and ventilation control is transfer learning, where a pre-trained neural network is fine-tuned with small amounts of measurement data from the target building [33]. In the context of RL, options include model-free RL, which demands extensive data, or model-based RL, which may inadequately capture physical dynamics, resulting in suboptimal performance in low-diversity data regimes [34]. For instance, in [35], an LSTM-based environment model trained on simulated data is used by the RL agent, but the model lacks physical insights into the building's thermal dynamics. Another technique is Dyna architecture, a special type of RL combining both model-free and model-based RL techniques enabling sample efficient RL learning with dynamic model-learning of the underlying system [36]. Gao and Wang [37] demonstrated that Dyna-style RL achieves higher sample efficiency and reduced training time compared to model-free alternatives for controlling building HVAC systems. However, in their study, the learned building model is purely data-driven, with no incorporation of physics, which limits its generalizability to unseen cases. This highlights the necessity for a physics-informed reinforcement learning (PIRL) approach that can address the challenges of low-diversity data regimes in building heating control, while leveraging both model-free and model-based RL techniques to provide a robust control solution.


---
Inspiration- OPTIMISE OB + ENERGY USE
