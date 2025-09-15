1. Concerning measurement, there is a need to collect data regularly under different operation states (i.e. failure or success) of the control modes. 

The data can be used to update the component performance model and improve the accuracy of risk/benefit evaluation. For instance, in the selected test case, if aggressive control fails, the outdoor air flowrate induced in practice can be obtained from actual measurements. The flowrate measurements can then be used for quantifying the energy waste of the aggressive mode.


2. Concerning the model prediction, there is a need for sensitivity and uncertainty analysis to identify the impacts of errors of different sensor on the *performance of the control strategies*. This would *reduce the need for data collection*, and simplify the risk/benefit analysis, hence facilitating online decision-making.

3. The successful implementation of the risk-based control strategy requires the predictive models to be adaptive to the changes of working conditions. 

4. The cost function (for evaluating risk and benefit) for decisionmaking must be modified for cleanrooms requiring higher cleanliness levels. 


**Balance**
In real applications, due to various uncertainties, engineers usually prefer to select a conservative/safe mode rather than an aggressive mode (i.e. with more energy-saving potential but higher risks), to ensure highly reliable operation.

@article{teng1996failure,
  title={Failure mode and effects analysis: an integrated approach for product design and process control},
  author={Teng, Sheng-Hsien and Ho, Shin-Yann},
  journal={International journal of quality \& reliability management},
  volume={13},
  number={5},
  pages={8--26},
  year={1996},
  publisher={MCB UP Ltd}
}

--
SMPC extends deterministic MPC by systematically incorporating uncertainty into the control framework. Uncontrollable inputs and measurement noise are modeled as random variables characterized by forecast-based probability distributions. A Kalman filter is employed to update state and output estimates along with their associated variances, ensuring that uncertainty is explicitly tracked over time. This formulation enables risk-aware, probabilistic decision-making, in contrast to deterministic MPC, which assumes perfect knowledge of system dynamics and inputs.

