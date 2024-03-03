Basic implementations of Physics Informed Neural Networks (PINN) with PyTorch.

PINN is a machine learning framework that integrates deep neural networks with physical laws or constraints, involving the solution of complex inverse problems and partial differential equations (PDEs). These combine domain knowledge with standard techniques and offer a powerful approach for solving inverse problems, identifying parameters, and predicting physical systems' behavior. These networks are trained on both observed data and governing equations, facilitating accurate predictions while ensuring adherence to underlying physics principles. PINNs find applications across various domains, including fluid dynamics, solid mechanics, and medical imaging, providing a versatile tool for scientific discovery and engineering applications.

I have implemented a PINN on the Burgers' Equation as discussed in the article linked below. I also provided an option to track the losses with using the residual term or not, which can be adjusted by the parameter `physics`.

The Burgers' Equation is as follows : 

   $u_t + u \cdot u_x - \left(\frac{0.01}{\pi}\right) u_{xx} = 0$
  
  ### Boundary Conditions
  
   $u(t, -1) = u(t, 1) = 0$
  
  ### Initial Condition
  
   $u(0, x) = -\sin(\pi x)$

![newplot](https://github.com/HridayM25/Physics-Informed-NN/assets/107138441/85ea7ea7-93ea-45d4-9b01-25a7aa2d916a)



# References 

Link: https://github.com/maziarraissi/PINNs/tree/master/appendix/continuous_time_identification%20(Burgers)

@article{raissi2017physicsI, title={Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations}, author={Raissi, Maziar and Perdikaris, Paris and Karniadakis, George Em}, journal={arXiv preprint arXiv:1711.10561}, year={2017} }

@article{raissi2017physicsII, title={Physics Informed Deep Learning (Part II): Data-driven Discovery of Nonlinear Partial Differential Equations}, author={Raissi, Maziar and Perdikaris, Paris and Karniadakis, George Em}, journal={arXiv preprint arXiv:1711.10566}, year={2017} }

Also credits to : https://github.com/jayroxis/PINNs/tree/master for forming the baseline for the code.
