BrainPy Examples
================

This repository contains examples of using `BrainPy <https://brainpy.readthedocs.io/>`_
to implement various models about neurons, synapse, networks, etc. We welcome your implementation,
which can be post through our `github <https://github.com/brainpy/examples>`_ page.

If you run some codes failed, please tell us through github issue https://github.com/brainpy/examples/issues .

If you found these examples are useful for your research, please kindly `cite us <https://brainpy.readthedocs.io/en/latest/tutorial_FAQs/citing_and_publication.html>`_.

If you want to add more examples, please fork our github https://github.com/brainpy/examples .



Example categories:

.. contents::
    :local:
    :depth: 2




Neuron Models
-------------

- `(Izhikevich, 2003): Izhikevich Model <neurons/Izhikevich_2003_Izhikevich_model.ipynb>`_
- `(Brette, Romain. 2004): LIF phase locking <neurons/Romain_2004_LIF_phase_locking.ipynb>`_
- `(Gerstner, 2005): Adaptive Exponential Integrate-and-Fire model <neurons/Gerstner_2005_AdExIF_model.ipynb>`_
- `(Niebur, et. al, 2009): Generalized integrate-and-fire model <neurons/Niebur_2009_GIF.ipynb>`_
- `(Jansen & Rit, 1995): Jansen-Rit Model <neurons/JR_1995_jansen_rit_model.ipynb>`_
- `(Teka, et. al, 2018): Fractional-order Izhikevich neuron model <neurons/2018_Fractional_Izhikevich_model.ipynb>`_
- `(Mondal, et. al, 2019): Fractional-order FitzHugh-Rinzel bursting neuron model <neurons/2019_Fractional_order_FHR_model.ipynb>`_



Continuous-attractor Network
----------------------------

- `CANN 1D Oscillatory Tracking <cann/Mi_2014_CANN_1D_oscillatory_tracking.ipynb>`_
- `(Si Wu, 2008): Continuous-attractor Neural Network 1D <cann/Wu_2008_CANN.ipynb>`_
- `(Si Wu, 2008): Continuous-attractor Neural Network 2D <cann/Wu_2008_CANN_2D.ipynb>`_



Decision Making Model
---------------------

- `(Wang, 2002): Decision making spiking model <decision_making/Wang_2002_decision_making_spiking.ipynb>`_
- `(Wong & Wang, 2006): Decision making rate model <decision_making/Wang_2006_decision_making_rate.ipynb>`_




E/I Balanced Network
--------------------


- `(Vreeswijk & Sompolinsky, 1996): E/I balanced network <ei_nets/Vreeswijk_1996_EI_net.ipynb>`_
- `(Brette, et, al., 2007): COBA <ei_nets/Brette_2007_COBA.ipynb>`_
- `(Brette, et, al., 2007): CUBA <ei_nets/Brette_2007_CUBA.ipynb>`_
- `(Brette, et, al., 2007): COBA-HH <ei_nets/Brette_2007_COBAHH.ipynb>`_
- `(Tian, et al., 2020): E/I Net for fast response <ei_nets/Tian_2020_EI_net_for_fast_response.ipynb>`_



Brain-inspired Computing
------------------------


- `Classify MNIST dataset by a fully connected LIF layer <https://github.com/brainpy/examples/blob/main/brain_inspired_computing/mnist_lif_readout.py>`_
- `Convolutional SNN to Classify Fashion-MNIST <https://github.com/brainpy/examples/blob/main/brain_inspired_computing/fashion_mnist_conv_lif.py>`_
- `(2022, NeurIPS): Online Training Through Time for Spiking Neural Networks <https://github.com/brainpy/examples/blob/main/brain_inspired_computing/OTTT-SNN.py>`_
- `(2019, Zenke, F.): SNN Surrogate Gradient Learning <https://github.com/brainpy/examples/blob/main/brain_inspired_computing/SurrogateGrad_lif-ANN.py>`_
- `(2019, Zenke, F.): SNN Surrogate Gradient Learning to Classify Fashion-MNIST <https://github.com/brainpy/examples/blob/main/brain_inspired_computing/SurrogateGrad_lif_fashion_mnist.py>`_
- `(2021, Raminmh): Liquid time-constant Networks <https://github.com/brainpy/examples/blob/main/brain_inspired_computing/liquid_time_constant_network.py>`_



Reservoir Computing
-------------------


- `Predicting Mackey-Glass timeseries <reservoir_computing/predicting_Mackey_Glass_timeseries.ipynb>`_
- `(Sussillo & Abbott, 2009): FORCE Learning <recurrent_networks/Sussillo_Abbott_2009_FORCE_Learning.ipynb>`_
- `(Gauthier, et. al, 2021): Next generation reservoir computing <reservoir_computing/Gauthier_2021_ngrc.ipynb>`_



Gap Junction Network
--------------------

- `(Fazli and Richard, 2022): Electrically Coupled Bursting Pituitary Cells <gj_nets/Fazli_2022_gj_coupled_bursting_pituitary_cells.ipynb>`_
- `(Sherman & Rinzel, 1992): Gap junction leads to anti-synchronization <gj_nets/Sherman_1992_gj_antisynchrony.ipynb>`_



Oscillation and Synchronization
-------------------------------

- `(Wang & Buzsáki, 1996): Gamma Oscillation <oscillation_synchronization/Wang_1996_gamma_oscillation.ipynb>`_
- `(Brunel & Hakim, 1999): Fast Global Oscillation <oscillation_synchronization/Brunel_Hakim_1999_fast_oscillation.ipynb>`_
- `(Diesmann, et, al., 1999): Synfire Chains <oscillation_synchronization/Diesmann_1999_synfire_chains.ipynb>`_
- `(Li, et. al, 2017): Unified Thalamus Oscillation Model <oscillation_synchronization/Li_2017_unified_thalamus_oscillation_model.ipynb>`_
- `(Susin & Destexhe, 2021): Asynchronous Network <oscillation_synchronization/Susin_Destexhe_2021_gamma_oscillation_AI.ipynb>`_
- `(Susin & Destexhe, 2021): CHING Network for Generating Gamma Oscillation <oscillation_synchronization/Susin_Destexhe_2021_gamma_oscillation_CHING.ipynb>`_
- `(Susin & Destexhe, 2021): ING Network for Generating Gamma Oscillation <oscillation_synchronization/Susin_Destexhe_2021_gamma_oscillation_ING.ipynb>`_
- `(Susin & Destexhe, 2021): PING Network for Generating Gamma Oscillation <oscillation_synchronization/Susin_Destexhe_2021_gamma_oscillation_PING.ipynb>`_



Large-Scale Modeling
--------------------

- `(Joglekar, et. al, 2018): Inter-areal Balanced Amplification Figure 1 <large_scale_modeling/Joglekar_2018_InterAreal_Balanced_Amplification_figure1.ipynb>`_
- `(Joglekar, et. al, 2018): Inter-areal Balanced Amplification Figure 2 <large_scale_modeling/Joglekar_2018_InterAreal_Balanced_Amplification_figure2.ipynb>`_
- `(Joglekar, et. al, 2018): Inter-areal Balanced Amplification Figure 5 <large_scale_modeling/Joglekar_2018_InterAreal_Balanced_Amplification_figure5.ipynb>`_
- `Simulating 1-million-neuron networks with 1GB GPU memory <large_scale_modeling/EI_net_with_1m_neurons.ipynb>`_



Recurrent Neural Network
------------------------


- `(Sussillo & Abbott, 2009): FORCE Learning <recurrent_networks/Sussillo_Abbott_2009_FORCE_Learning.ipynb>`_
- `Integrator RNN Model <recurrent_networks/integrator_rnn.ipynb>`_
- `Train RNN to Solve Parametric Working Memory <recurrent_networks/ParametricWorkingMemory.ipynb>`_
- `(Song, et al., 2016): Training excitatory-inhibitory recurrent network <recurrent_networks/Song_2016_EI_RNN.ipynb>`_
- `(Masse, et al., 2019): RNN with STP for Working Memory  <recurrent_networks/Masse_2019_STP_RNN.ipynb>`_
- `(Yang, 2020): Dynamical system analysis for RNN <recurrent_networks/Yang_2020_RNN_Analysis.ipynb>`_
- `(Bellec, et. al, 2020): eprop for Evidence Accumulation Task <recurrent_networks/Bellec_2020_eprop_evidence_accumulation.ipynb>`_



Working Memory Model
--------------------

- `(Bouchacourt & Buschman, 2019): Flexible Working Memory Model <working_memory/Bouchacourt_2019_Flexible_working_memory.ipynb>`_
- `(Mi, et. al., 2017): STP for Working Memory Capacity <working_memory/Mi_2017_working_memory_capacity.ipynb>`_
- `(Masse, et al., 2019): RNN with STP for Working Memory  <recurrent_networks/Masse_2019_STP_RNN.ipynb>`_



Dynamics Analysis
-----------------

- `[1D] Simple systems <dynamics_analysis/1d_simple_systems.ipynb>`_
- `[2D] NaK model analysis <dynamics_analysis/2d_NaK_model.ipynb>`_
- `[2D] Wilson-Cowan model <dynamics_analysis/2d_wilson_cowan_model.ipynb>`_
- `[2D] Decision Making Model with SlowPointFinder <dynamics_analysis/2d_decision_making_model.ipynb>`_
- `[2D] Decision Making Model with Low-dimensional Analyzer <dynamics_analysis/2d_decision_making_with_lowdim_analyzer.ipynb>`_
- `[3D] Hindmarsh Rose Model <dynamics_analysis/3d_hindmarsh_rose_model.ipynb>`_
- `Continuous-attractor Neural Network <dynamics_analysis/highdim_CANN.ipynb>`_
- `Gap junction-coupled FitzHugh-Nagumo Model <dynamics_analysis/highdim_gj_coupled_fhn.ipynb>`_
- `(Yang, 2020): Dynamical system analysis for RNN <recurrent_networks/Yang_2020_RNN_Analysis.ipynb>`_




Classical Dynamical Systems
---------------------------

- `Hénon map <classical_dynamical_systems/henon_map.ipynb>`_
- `Logistic map <classical_dynamical_systems/logistic_map.ipynb>`_
- `Lorenz system <classical_dynamical_systems/lorenz_system.ipynb>`_
- `Mackey-Glass equation <classical_dynamical_systems/mackey_glass_eq.ipynb>`_
- `Multiscroll chaotic attractor (多卷波混沌吸引子) <classical_dynamical_systems/Multiscroll_attractor.ipynb>`_
- `Rabinovich-Fabrikant equations <classical_dynamical_systems/Rabinovich_Fabrikant_eq.ipynb>`_
- `Fractional-order Chaos Gallery <classical_dynamical_systems/fractional_order_chaos.ipynb>`_





Unclassified Models
-------------------

- `(Brette & Guigon, 2003): Reliability of spike timing <others/Brette_Guigon_2003_spike_timing_reliability.ipynb>`_





Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
