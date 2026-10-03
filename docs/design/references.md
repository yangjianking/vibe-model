# 理论文献出处总表（VIBE-Model）

本表是全仓库公式与算法出处的**唯一索引**。设计文档与源码注释使用方括号编号引用，
例如 `[D3]`、`[T7]`、`[V4]`。

分组：

| 前缀 | 主题 |
|---|---|
| **B** | 教材与综述（背景） |
| **D** | 控制方程、垂直坐标、守恒离散、平流格式 |
| **T** | 时间离散、半隐式、声波子步、线性求解器 |
| **N** | 嵌套、侧边界、变分辨率网格 |
| **G** | GPU、混合精度、高性能计算 |
| **V** | 变分同化与 4D-Var |
| **A** | 切线性/伴随与自动微分 |
| **O** | 观测算子与资料 |
| **P** | 物理参数化 |
| **E** | 误差检验与预报评估 |

---

## B. 教材与综述

- [B1] Kalnay, E. (2003). *Atmospheric Modeling, Data Assimilation and Predictability*. Cambridge University Press.
- [B2] Durran, D. R. (2010). *Numerical Methods for Fluid Dynamics: With Applications to Geophysics* (2nd ed.). Springer.
- [B3] Warner, T. T. (2011). *Numerical Weather and Climate Prediction*. Cambridge University Press.
- [B4] Holton, J. R., & Hakim, G. J. (2013). *An Introduction to Dynamic Meteorology* (5th ed.). Academic Press.
- [B5] Vallis, G. K. (2017). *Atmospheric and Oceanic Fluid Dynamics* (2nd ed.). Cambridge University Press.
- [B6] LeVeque, R. J. (2002). *Finite Volume Methods for Hyperbolic Problems*. Cambridge University Press.
- [B7] Toro, E. F. (2009). *Riemann Solvers and Numerical Methods for Fluid Dynamics* (3rd ed.). Springer.
- [B8] Nocedal, J., & Wright, S. J. (2006). *Numerical Optimization* (2nd ed.). Springer.
- [B9] Wilks, D. S. (2019). *Statistical Methods in the Atmospheric Sciences* (4th ed.). Elsevier.
- [B10] Jolliffe, I. T., & Stephenson, D. B. (Eds.) (2012). *Forecast Verification: A Practitioner's Guide in Atmospheric Science* (2nd ed.). Wiley.
- [B11] Higham, N. J. (2002). *Accuracy and Stability of Numerical Algorithms* (2nd ed.). SIAM.
- [B12] Gropp, W., Lusk, E., & Skjellum, A. (1999). *Using MPI: Portable Parallel Programming with the Message-Passing Interface* (2nd ed.). MIT Press.

---

## D. 控制方程、垂直坐标、守恒离散与平流

- [D1] Klemp, J. B., & Wilhelmson, R. B. (1978). The simulation of three-dimensional convective storm dynamics. *Journal of the Atmospheric Sciences*, 35(6), 1070–1096.
- [D2] Gal-Chen, T., & Somerville, R. C. J. (1975). On the use of a coordinate transformation for the solution of the Navier–Stokes equations. *Journal of Computational Physics*, 17(2), 209–228.
- [D3] Arakawa, A., & Lamb, V. R. (1977). Computational design of the basic dynamical processes of the UCLA general circulation model. *Methods in Computational Physics*, 17, 173–265.
- [D4] Lorenz, E. N. (1960). Energy and numerical weather prediction. *Tellus*, 12(4), 364–373.
- [D5] Skamarock, W. C., & Klemp, J. B. (2008). A time-split nonhydrostatic atmospheric model for weather research and forecasting applications. *Journal of Computational Physics*, 227(7), 3465–3485.
- [D6] Klemp, J. B., Skamarock, W. C., & Dudhia, J. (2008). Conservative split-explicit time integration methods for the compressible nonhydrostatic equations. *Monthly Weather Review*, 136(8), 4648–4665.
- [D7] Shu, C.-W. (1998). Essentially non-oscillatory and weighted essentially non-oscillatory schemes for hyperbolic conservation laws. In *Advanced Numerical Approximation of Nonlinear Hyperbolic Equations*, Lecture Notes in Mathematics 1697, Springer, 325–432.
- [D8] Jiang, G.-S., & Shu, C.-W. (1996). Efficient implementation of weighted ENO schemes. *Journal of Computational Physics*, 126(1), 202–228.
- [D9] Schär, C., Leuenberger, D., Fuhrer, O., Lüthi, D., & Girard, C. (2002). A new terrain-following vertical coordinate formulation for atmospheric prediction models. *Monthly Weather Review*, 130(10), 2459–2480.
- [D10] Klemp, J. B. (2011). A terrain-following coordinate with smoothed coordinate surfaces. *Monthly Weather Review*, 139(7), 2163–2169.
- [D11] Zängl, G. (2012). Extending the numerical stability limit of terrain-following coordinate models over steep slopes. *Monthly Weather Review*, 140(11), 3722–3733.
- [D12] Tripoli, G. J., & Cotton, W. R. (1982). The Colorado State University three-dimensional cloud/mesoscale model—1982. Part I: General theoretical framework and sensitivity experiments. *Journal de Recherches Atmospheriques*, 16, 185–219.
- [D13] Satoh, M. (2002). Conservative scheme for the compressible nonhydrostatic models in Cartesian and spherical coordinates. *Monthly Weather Review*, 130(5), 1223–1242.
- [D14] Harris, L. M., & Durran, D. R. (2010). A comparison of two-layer and continuous formulations of terrain-following coordinates. *Monthly Weather Review*, 138(9), 3568–3573.
- [D15] Baldauf, M., Seifert, A., Förstner, J., Majewski, D., Raschendorfer, M., & Reinhardt, T. (2011). Operational convective-scale numerical weather prediction with the COSMO model: Description and sensitivities. *Monthly Weather Review*, 139(12), 3887–3905.
- [D16] Skamarock, W. C., Klemp, J. B., Dudhia, J., Gill, D. O., Barker, D. M., Duda, M. G., Huang, X.-Y., Wang, W., & Powers, J. G. (2008). *A Description of the Advanced Research WRF Version 3*. NCAR Technical Note NCAR/TN-475+STR.

---

## T. 时间离散、半隐式、声波子步与线性求解器

- [T1] Kwizak, M., & Robert, A. J. (1971). A semi-implicit scheme for grid point atmospheric models of the primitive equations. *Monthly Weather Review*, 99(1), 32–36.
- [T2] Tapp, M. C., & White, P. W. (1976). A non-hydrostatic mesoscale model. *Quarterly Journal of the Royal Meteorological Society*, 102(432), 277–296.
- [T3] Robert, A. (1982). A semi-Lagrangian and semi-implicit numerical integration scheme for the primitive meteorological equations. *Journal of the Meteorological Society of Japan*, 60(1), 319–325.
- [T4] Staniforth, A., & Côté, J. (1991). Semi-Lagrangian integration schemes for atmospheric models — A review. *Monthly Weather Review*, 119(9), 2206–2223.
- [T5] Wicker, L. J., & Skamarock, W. C. (2002). Time-splitting methods for elastic models using forward time schemes. *Monthly Weather Review*, 130(8), 2088–2097.
- [T6] Cullen, M. J. P. (1990). A test of a semi-implicit non-hydrostatic mesoscale model. *Quarterly Journal of the Royal Meteorological Society*, 116(496), 1663–1674.
- [T7] Bénard, P. (2003). Stability of semi-implicit and iterative centered-implicit time discretizations for various equation systems used in NWP. *Monthly Weather Review*, 131(10), 2479–2491.
- [T8] Bénard, P. (2004). On the use of a wider class of linear systems for the design of constant-coefficients semi-implicit time schemes in NWP. *Monthly Weather Review*, 132(6), 1319–1324.
- [T9] Satoh, M., Matsuno, T., Tomita, H., Miura, H., Nasuno, T., & Iga, S. (2008). Nonhydrostatic icosahedral atmospheric model (NICAM) for global cloud resolving simulations. *Journal of Computational Physics*, 227(7), 3486–3514.
- [T10] Thomas, L. H. (1949). *Elliptic Problems in Linear Difference Equations over a Network*. Watson Scientific Computing Laboratory Report, Columbia University.
- [T11] Saad, Y., & Schultz, M. H. (1986). GMRES: A generalized minimal residual algorithm for solving nonsymmetric linear systems. *SIAM Journal on Scientific and Statistical Computing*, 7(3), 856–869.
- [T12] van der Vorst, H. A. (1992). Bi-CGSTAB: A fast and smoothly converging variant of Bi-CG for the solution of nonsymmetric linear systems. *SIAM Journal on Scientific and Statistical Computing*, 13(2), 631–644.
- [T13] Briggs, W. L., Henson, V. E., & McCormick, S. F. (2000). *A Multigrid Tutorial* (2nd ed.). SIAM.
- [T14] Wood, N., Staniforth, A., White, A., Allen, T., Diamantakis, M., Gross, M., Melvin, T., Smith, C., Vosper, S., Zerroukat, M., & Thuburn, R. (2014). An inherently mass-conserving semi-implicit semi-Lagrangian discretization of the deep-atmosphere global non-hydrostatic equations. *Quarterly Journal of the Royal Meteorological Society*, 140(682), 1505–1520.
- [T15] Baldauf, M. (2010). Linear stability analysis of Runge–Kutta-based partial time-splitting schemes for the Euler equations. *Monthly Weather Review*, 138(12), 4475–4496.
- [T16] Williamson, D. L. (1980). *Difference Approximations for Fluid Flow on the Sphere*. NCAR Technical Note.
- [T17] Leonard, B. P. (1979). A stable and accurate convective modelling procedure based on quadratic upstream interpolation. *Computer Methods in Applied Mechanics and Engineering*, 19(1), 59–98.
- [T18] Rai, M. M., & Moin, P. (1991). Direct simulations of turbulent flow using finite-difference schemes. *Journal of Computational Physics*, 96(1), 15–53.
- [T19] Zängl, G., Reinert, D., Rípodas, P., & Baldauf, M. (2015). The ICON (ICOsahedral Non-hydrostatic) modelling framework of DWD and MPI-M. *Quarterly Journal of the Royal Meteorological Society*, 141(687), 563–579.

---

## N. 嵌套、侧边界与变分辨率网格

- [N1] Davies, H. C. (1976). A lateral boundary formulation for multi-level prediction models. *Quarterly Journal of the Royal Meteorological Society*, 102(432), 405–418.
- [N2] Davies, H. C. (1983). Limitations of some common lateral boundary schemes used in regional NWP models. *Monthly Weather Review*, 111(5), 1002–1012.
- [N3] Clark, T. L., & Farley, R. D. (1984). Severe downslope windstorm calculations in two and three spatial dimensions using anelastic interactive grid nesting: A possible mechanism for gustiness. *Journal of the Atmospheric Sciences*, 41(3), 329–350.
- [N4] Skamarock, W. C., & Klemp, J. B. (1993). Adaptive grid refinement for two-dimensional and three-dimensional nonhydrostatic atmospheric flow. *Monthly Weather Review*, 121(3), 788–804.
- [N5] Skamarock, W. C., Klemp, J. B., Duda, M. G., Fowler, L. D., Park, S.-H., & Ringler, T. D. (2012). A multiscale nonhydrostatic atmospheric model using centroidal Voronoi tessellations and C-grid staggering. *Monthly Weather Review*, 140(9), 3090–3105.
- [N6] Ringler, T., Petersen, M., Higdon, R. L., Jacobsen, D., Jones, P. W., & Maltrud, M. (2013). A multi-resolution approach to global ocean modeling. *Ocean Modelling*, 69, 211–232.
- [N7] Tomita, H., & Satoh, M. (2004). A new dynamical framework of nonhydrostatic global model using the icosahedral grid. *Fluid Dynamics Research*, 34(6), 357–400.
- [N8] Harris, L. M., & Lin, S.-J. (2013). A two-way nested global-regional dynamical core on the cubed-sphere grid. *Monthly Weather Review*, 141(1), 283–306.
- [N9] Warner, T. T., Peterson, R. A., & Treadon, R. E. (1997). A tutorial on lateral boundary conditions as a basic and potentially serious limitation to regional numerical weather prediction. *Bulletin of the American Meteorological Society*, 78(11), 2599–2617.
- [N10] Koch, S. E., DesJardins, M., & Kocin, P. J. (1983). A nested grid initialization procedure. *Monthly Weather Review*, 111(12), 2473–2488.

---

## G. GPU、混合精度与高性能计算

- [G1] Micikevicius, P., Narang, S., Alben, J., Diamos, G., Elsen, E., Garcia, D., Ginsburg, B., Houston, M., Kuchaiev, O., Venkatesh, G., & Wu, H. (2018). Mixed precision training. *International Conference on Learning Representations (ICLR)*. arXiv:1710.03740.
- [G2] Kahan, W. (1965). Further remarks on reducing truncation errors. *Communications of the ACM*, 8(1), 40.
- [G3] Neumaier, A. (1974). Rundungsfehleranalyse einiger Verfahren zur Summation endlicher Summen. *Zeitschrift für Angewandte Mathematik und Mechanik*, 54(1), 39–51.
- [G4] Ogita, T., Rump, S. M., & Oishi, S. (2005). Accurate sum and dot product. *SIAM Journal on Scientific Computing*, 26(6), 1955–1988.
- [G5] Haidar, A., Tomov, S., Dongarra, J., & Higham, N. J. (2018). Harnessing GPU tensor cores for fast FP16 arithmetic to speed up mixed-precision iterative refinement solvers. *Proceedings of SC18*.
- [G6] Klöwer, M., Hatfield, S., Croci, M., Düben, P. D., & Palmer, T. N. (2019). Mixed-precision arithmetic in the ENDGame dynamical core of the Unified Model. *Geoscientific Model Development*, 12, 4425–4447.
- [G7] Düben, P. D., & Palmer, T. N. (2014). Benchmarking of numerical precision in weather and climate models. *Journal of Computational Physics*, 278, 1–12.
- [G8] Dawson, A., & Düben, P. D. (2017). rpe: A numpy-based emulation framework for reduced precision. *Procedia Computer Science*, 108, 1450–1459.
- [G9] Bauer, P., Dueben, P. D., Hoefler, T., Quintino, T., Schulthess, T. C., & Wedi, N. P. (2021). The digital revolution of Earth-system science. *Nature Computational Science*, 1(2), 104–113.
- [G10] NVIDIA Corporation (2023). *CUDA C++ Programming Guide*. NVIDIA Developer Documentation.
- [G11] Khronos Group (2021). *SYCL 2020 Specification*.
- [G12] AMD (2023). *HIP Programming Guide (ROCm Documentation)*.
- [G13] OpenMP Architecture Review Board (2021). *OpenMP Application Programming Interface, Version 5.2*.
- [G14] Message Passing Interface Forum (2021). *MPI: A Message-Passing Interface Standard, Version 4.0*.
- [G15] Cerveny, R. S. (2019)? — (保留占位，未引用)
- [G16] Stone, J. E. (1973). Iterative solution of implicit approximations of multidimensional partial differential equations. *SIAM Journal on Numerical Analysis*, 5(3), 530–558.
- [G17] Gustafson, J. L. (1988). Reevaluating Amdahl's law. *Communications of the ACM*, 31(5), 532–533.
- [G18] Dennis, J. B., & Edwards, H. C. (2004). *An Updated Checklist for Choosing a High-Performance Computing Platform*. NCAR Technical Note.

---

## V. 变分同化与 4D-Var

- [V1] Le Dimet, F.-X., & Talagrand, O. (1986). Variational algorithms for analysis and assimilation of meteorological observations: Theoretical aspects. *Tellus A*, 38(2), 97–110.
- [V2] Talagrand, O., & Courtier, P. (1987). Variational assimilation of meteorological observations with the adjoint vorticity equation. I: Theory. *Quarterly Journal of the Royal Meteorological Society*, 113(478), 1311–1328.
- [V3] Courtier, P., Thépaut, J.-N., & Hollingsworth, A. (1994). A strategy for operational implementation of 4D-Var, using an incremental approach. *Quarterly Journal of the Royal Meteorological Society*, 120(519), 1367–1387.
- [V4] Lorenc, A. C. (1986). Analysis methods for numerical weather prediction. *Quarterly Journal of the Royal Meteorological Society*, 112(474), 1177–1194.
- [V5] Parrish, D. F., & Derber, J. C. (1992). The National Meteorological Center's spectral statistical-interpolation analysis system. *Monthly Weather Review*, 120(8), 1747–1763.
- [V6] Derber, J., & Rosati, A. (1989). A global oceanic data assimilation system. *Journal of Physical Oceanography*, 19(9), 1333–1347.
- [V7] Weaver, A., & Courtier, P. (2001). Correlation modelling on the sphere using a generalized diffusion equation. *Quarterly Journal of the Royal Meteorological Society*, 127(575), 1815–1846.
- [V8] Bannister, R. N. (2008). A review of forecast error covariance statistics in atmospheric variational data assimilation. I: Characteristics and measurements of forecast error covariances. *Quarterly Journal of the Royal Meteorological Society*, 134(637), 1951–1970.
- [V9] Bannister, R. N. (2008). A review of forecast error covariance statistics in atmospheric variational data assimilation. II: Modelling the forecast error covariance statistics. *Quarterly Journal of the Royal Meteorological Society*, 134(637), 1971–1996.
- [V10] Rabier, F., Järvinen, H., Klinker, E., Mahfouf, J.-F., & Simmons, A. (2000). The ECMWF operational implementation of four-dimensional variational assimilation. I: Experimental results with simplified physics. *Quarterly Journal of the Royal Meteorological Society*, 126(564), 1143–1170.
- [V11] Rawlins, F., Ballard, S. P., Bovis, K. J., Clayton, A. M., Li, D., Inverarity, G. W., Lorenc, A. C., & Payne, T. J. (2007). The Met Office global four-dimensional variational data assimilation scheme. *Quarterly Journal of the Royal Meteorological Society*, 133(623), 347–362.
- [V12] Fisher, M., & Courtier, P. (1995). *Estimating the Covariance Matrices of Analysis and Forecast Error in Variational Data Assimilation*. ECMWF Technical Memorandum 220.
- [V13] Liu, D. C., & Nocedal, J. (1989). On the limited memory BFGS method for large scale optimization. *Mathematical Programming*, 45(1), 503–528.
- [V14] Ide, K., Courtier, P., Ghil, M., & Lorenc, A. C. (1997). Unified notation for data assimilation: Operational, sequential and variational. *Journal of the Meteorological Society of Japan*, 75(1B), 181–189.
- [V15] Trémolet, Y. (2007). Model-error estimation in 4D-Var. *Quarterly Journal of the Royal Meteorological Society*, 133(626), 1267–1280.
- [V16] Dee, D. P. (2005). Bias and data assimilation. *Quarterly Journal of the Royal Meteorological Society*, 131(613), 3323–3343.
- [V17] Auligné, T., McNally, A. P., & Dee, D. P. (2007). Adaptive bias correction for satellite data in a numerical weather prediction system. *Quarterly Journal of the Royal Meteorological Society*, 133(624), 631–642.
- [V18] Desroziers, G., Berre, L., Chapnik, B., & Poli, P. (2005). Diagnosis of observation, background and analysis-error statistics in observation space. *Quarterly Journal of the Royal Meteorological Society*, 131(613), 3385–3396.
- [V19] Evensen, G. (1994). Sequential data assimilation with a nonlinear quasi-geostrophic model using Monte Carlo methods to forecast error statistics. *Journal of Geophysical Research*, 99(C5), 10143–10162.
- [V20] Hamill, T. M., & Snyder, C. (2000). A hybrid ensemble Kalman filter–3D variational analysis scheme. *Monthly Weather Review*, 128(8), 2905–2919.
- [V21] Wang, X., Bishop, C. H., & Julier, S. J. (2004). Which is better, an ensemble of positive–negative pairs or a centered spherical simplex ensemble? *Monthly Weather Review*, 132(7), 1590–1605.
- [V22] Lawless, A. S., Gratton, S., & Nichols, N. K. (2006). An investigation of incremental 4D-Var using non-tangent linear models. *Quarterly Journal of the Royal Meteorological Society*, 132(619), 1541–1560.
- [V23] Lorenc, A. C. (2003). The potential of the ensemble Kalman filter for NWP — a comparison with 4D-Var. *Quarterly Journal of the Royal Meteorological Society*, 129(595), 3183–3203.
- [V24] Thépaut, J.-N., & Courtier, P. (1991). Four-dimensional variational data assimilation using the adjoint of a multilevel primitive-equation model. *Quarterly Journal of the Royal Meteorological Society*, 117(502), 1225–1254.
- [V25] Gauthier, P., & Thépaut, J.-N. (2001). Impact of the digital filter as a weak constraint in the preoperational 4DVAR assimilation system of Météo-France. *Monthly Weather Review*, 129(8), 2089–2102.

---

## A. 切线性、伴随与自动微分

- [A1] Giering, R., & Kaminski, T. (1998). Recipes for adjoint code construction. *ACM Transactions on Mathematical Software*, 24(4), 437–474.
- [A2] Griewank, A. (2000). *Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation*. SIAM.
- [A3] Sirkes, Z., & Tziperman, E. (1997). Finite difference of adjoint or adjoint of finite difference? *Monthly Weather Review*, 125(12), 3373–3378.
- [A4] Errico, R. M. (1997). What is an adjoint model? *Bulletin of the American Meteorological Society*, 78(11), 2577–2591.
- [A5] Navon, I. M., Zou, X., Derber, J., & Sela, J. (1992). Variational data assimilation with an adiabatic version of the NMC spectral model. *Monthly Weather Review*, 120(7), 1433–1446.
- [A6] Zou, X., Navon, I. M., & Sela, J. (1993). Variational data assimilation with moist threshold processes using the NMC spectral model. *Tellus A*, 45(5), 370–387.
- [A7] Courtier, P., & Talagrand, O. (1987). Variational assimilation of meteorological observations with the adjoint vorticity equation. II: Numerical results. *Quarterly Journal of the Royal Meteorological Society*, 113(478), 1329–1347.
- [A8] Mahfouf, J.-F. (1999). Influence of physical processes on the tangent-linear approximation. *Tellus A*, 51(2), 147–166.
- [A9] Zhu, Y., & Gelaro, R. (2008). Observation sensitivity calculations using the adjoint of the Gridpoint Statistical Interpolation (GSI) analysis system. *Monthly Weather Review*, 136(1), 335–351.
- [A10] Walther, A., & Griewank, A. (2009). Getting started with ADOL-C. In *Combinatorial Scientific Computing*, Chapman & Hall/CRC.

---

## O. 观测算子与观测资料

- [O1] Saunders, R., Matricardi, M., & Brunel, P. (1999). An improved fast radiative transfer model for assimilation of satellite radiance observations. *Quarterly Journal of the Royal Meteorological Society*, 125(556), 1407–1425.
- [O2] Hocking, J., Rayer, P., Rundle, D., Saunders, R., Matricardi, M., Geer, A., Brunel, P., & Vidot, J. (2013). *RTTOV v11 Users Guide*. EUMETSAT NWP-SAF Report.
- [O3] Eyre, J. R. (1990). The information content of data from satellite sounding systems: A simulation study. *Quarterly Journal of the Royal Meteorological Society*, 116(492), 401–434.
- [O4] Healy, S. B., & Eyre, J. R. (2000). Retrieving temperature, water vapour and surface pressure information from refractive-index profiles derived by radio occultation: A simulation study. *Quarterly Journal of the Royal Meteorological Society*, 126(566), 1661–1683.
- [O5] Kuo, Y.-H., Wee, T.-K., Sokolovskiy, S., Rocken, C., Schreiner, W., Hunt, D., & Anthes, R. A. (2004). Inversion and error estimation of GPS radio occultation data. *Journal of the Meteorological Society of Japan*, 82(1B), 507–531.
- [O6] Sun, J., & Crook, N. A. (1997). Dynamical and microphysical retrieval from Doppler radar observations using a cloud model and its adjoint. Part I: Model development and simulated data experiments. *Journal of the Atmospheric Sciences*, 54(12), 1642–1661.
- [O7] Velden, C. S., Daniels, J., Stettner, D., Santek, D., Key, J., Dunion, J., Holmlund, K., Dengel, G., Bresky, W., & Menzel, P. (2005). Recent innovations in deriving tropospheric winds from meteorological satellites. *Bulletin of the American Meteorological Society*, 86(2), 205–223.
- [O8] Lorenc, A. C., Ballard, S. P., Bell, R. S., Ingleby, N. B., Andrews, P. L. F., Barker, D. M., Bray, J. R., Clayton, A. M., Dalby, T., Li, D., Payne, T. J., & Saunders, F. W. (2000). The Met. Office global three-dimensional variational data assimilation scheme. *Quarterly Journal of the Royal Meteorological Society*, 126(570), 2991–3012.
- [O9] WMO (2018). *Guide to Instruments and Methods of Observation (WMO-No. 8)*. World Meteorological Organization.
- [O10] Rodgers, C. D. (2000). *Inverse Methods for Atmospheric Sounding: Theory and Practice*. World Scientific.

---

## P. 物理参数化

- [P1] Kessler, E. (1969). On the distribution and continuity of water substance in atmospheric circulations. *Meteorological Monographs*, 10(32), 1–84.
- [P2] Lin, Y.-L., Farley, R. D., & Orville, H. D. (1983). Bulk parameterization of the snow field in a cloud model. *Journal of Climate and Applied Meteorology*, 22(6), 1065–1092.
- [P3] Thompson, G., Field, P. R., Rasmussen, R. M., & Hall, W. D. (2008). Explicit forecasts of winter precipitation using an improved bulk microphysics scheme. Part II: Implementation of a new snow parameterization. *Monthly Weather Review*, 136(12), 5095–5115.
- [P4] Morrison, H., Curry, J. A., & Khvorostyanov, V. I. (2005). A new double-moment microphysics parameterization for application in cloud and climate models. Part I: Description. *Journal of the Atmospheric Sciences*, 62(6), 1665–1677.
- [P5] Mlawer, E. J., Taubman, S. J., Brown, P. D., Iacono, M. J., & Clough, S. A. (1997). Radiative transfer for inhomogeneous atmospheres: RRTM, a validated correlated-k model for the longwave. *Journal of Geophysical Research*, 102(D14), 16663–16682.
- [P6] Iacono, M. J., Delamere, J. S., Mlawer, E. J., Shephard, M. W., Clough, S. A., & Collins, W. D. (2008). Radiative forcing by long-lived greenhouse gases: Calculations with the AER radiative transfer models. *Journal of Geophysical Research*, 113, D13103.
- [P7] Hong, S.-Y., Noh, Y., & Dudhia, J. (2006). A new vertical diffusion package with an explicit treatment of entrainment processes. *Monthly Weather Review*, 134(9), 2318–2341.
- [P8] Mellor, G. L., & Yamada, T. (1982). Development of a turbulence closure model for geophysical fluid problems. *Reviews of Geophysics*, 20(4), 851–875.
- [P9] Monin, A. S., & Obukhov, A. M. (1954). Basic laws of turbulent mixing in the surface layer of the atmosphere. *Trudy Akademii Nauk SSSR Geofizicheskii Institut*, 24(151), 163–187.
- [P10] Businger, J. A., Wyngaard, J. C., Izumi, Y., & Bradley, E. F. (1971). Flux-profile relationships in the atmospheric surface layer. *Journal of the Atmospheric Sciences*, 28(2), 181–189.
- [P11] Kain, J. S., & Fritsch, J. M. (1990). A one-dimensional entraining/detraining plume model and its application in convective parameterization. *Journal of the Atmospheric Sciences*, 47(23), 2784–2802.
- [P12] Grell, G. A., & Dévényi, D. (2002). A generalized approach to parameterizing convection combining ensemble and data assimilation techniques. *Geophysical Research Letters*, 29(14), 1693.
- [P13] Tiedtke, M. (1989). A comprehensive mass flux scheme for cumulus parameterization in large-scale models. *Monthly Weather Review*, 117(8), 1779–1800.
- [P14] Noilhan, J., & Planton, S. (1989). A simple parameterization of land surface processes for meteorological models. *Monthly Weather Review*, 117(3), 536–549.
- [P15] Chen, F., & Dudhia, J. (2001). Coupling an advanced land surface–hydrology model with the Penn State–NCAR MM5 modeling system. Part I: Model implementation and sensitivity. *Monthly Weather Review*, 129(4), 569–585.
- [P16] Smagorinsky, J. (1963). General circulation experiments with the primitive equations: I. The basic experiment. *Monthly Weather Review*, 91(3), 99–164.
- [P17] Deardorff, J. W. (1980). Stratocumulus-capped mixed layers derived from a three-dimensional model. *Boundary-Layer Meteorology*, 18(4), 495–527.
- [P18] Janjić, Z. I. (1994). The step-mountain eta coordinate model: Further developments of the convection, viscous sublayer, and turbulence closure schemes. *Monthly Weather Review*, 122(5), 927–945.

---

## E. 误差检验与预报评估

- [E1] Murphy, A. H. (1993). What is a good forecast? An essay on the nature of goodness in weather forecasting. *Weather and Forecasting*, 8(2), 281–293.
- [E2] Murphy, A. H., & Winkler, R. L. (1987). A general framework for forecast verification. *Monthly Weather Review*, 115(7), 1330–1338.
- [E3] Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. *Monthly Weather Review*, 78(1), 1–3.
- [E4] Gandin, L. S., & Murphy, A. H. (1992). Equitable skill scores for categorical forecasts. *Monthly Weather Review*, 120(2), 361–370.
- [E5] Hanssen, A. W., & Kuipers, W. J. A. (1965). On the relationship between the frequency of rain and various meteorological parameters. *Mededelingen en Verhandelingen*, 81, 2–15.
- [E6] Schaefer, J. T. (1990). The critical success index as an indicator of warning skill. *Weather and Forecasting*, 5(4), 570–575.
- [E7] Roberts, N. M., & Lean, H. W. (2008). Scale-selective verification of rainfall accumulations from high-resolution forecasts of convective events. *Monthly Weather Review*, 136(1), 78–97.
- [E8] Ebert, E. E. (2008). Fuzzy verification of high-resolution gridded forecasts: A review and proposed framework. *Meteorological Applications*, 15(1), 51–64.
- [E9] Hamill, T. M. (2001). Interpretation of rank histograms for verifying ensemble forecasts. *Monthly Weather Review*, 129(3), 550–560.
- [E10] Talagrand, O., & Vautard, R. (1997). Evaluation of probabilistic prediction systems. *Proceedings of the ECMWF Workshop on Predictability*, 1–25.
- [E11] Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *Journal of the American Statistical Association*, 102(477), 359–378.
- [E12] Hersbach, H. (2000). Decomposition of the continuous ranked probability score for ensemble prediction systems. *Weather and Forecasting*, 15(5), 559–570.
- [E13] Casati, B., Wilson, L. J., Stephenson, D. B., Nurmi, P., Ghelli, A., Pocernich, M., Damrath, U., Ebert, E. E., Brown, B. G., & Mason, S. (2008). Forecast verification: Current status and future directions. *Meteorological Applications*, 15(1), 3–18.
- [E14] Mittermaier, M., & Roberts, N. (2010). Intercomparison of spatial forecast verification methods: Identifying skillful spatial scales using the fractions skill score. *Weather and Forecasting*, 25(1), 291–301.
- [E15] Mason, S. J., & Graham, N. E. (2002). Areas beneath the relative operating characteristics (ROC) and relative operating levels (ROL) curves: Statistical significance and interpretation. *Quarterly Journal of the Royal Meteorological Society*, 128(584), 2145–2166.
- [E16] Ferro, C. A. T., Richardson, D. S., & Weigel, A. P. (2008). On the effect of ensemble size on the discrete and continuous ranked probability scores. *Meteorological Applications*, 15(1), 19–24.
- [E17] Brier, G. W., & Allen, R. A. (1951). Verification of weather forecasts. In *Compendium of Meteorology*, American Meteorological Society, 841–848.
- [E18] Stanski, H. R., Wilson, L. J., & Burrows, W. R. (1989). *Survey of Common Verification Methods in Meteorology*. WMO World Weather Watch Technical Report No. 8.
- [E19] WMO (2012). *Manual on the Global Data-Processing and Forecasting System (WMO-No. 485)*, Appendix II.4: Verification of forecasts.
