# Simulation-Guided Causal History and Motion-Aware Learning for Stable Soft Magnetic Tactile Trajectory Reconstruction

**Abstract** Continuous reconstruction with soft magnetic tactile sensors is affected by spatial encoding, deformation history, and magnetic observation disturbances. More training data or smoother-looking outputs need not improve localization. We develop an axis-separated dynamic forward surrogate and interactive simulation platform from source-device motions and fifteen-channel magnetic records. Controlled virtual motions motivate a causal representation that is subsequently refined using source-real supervision. Three exponential-average residuals are appended to an already optimized planar residual head, preserving its original histories, normalization, depth estimator, and output permission. Zero initialization of the additional connections reproduces the original function exactly. Two matched candidates use position supervision and combined position, motion-increment, qualified-dwell, and static-anchor supervision. Frozen causal replay evaluates 36 complex real trajectories excluded from fitting and checkpoint selection, preserving actual unloaded baselines, timestamps, and state lifecycles. On two source-device batches, the constrained candidate reduces raw XY error from 0.436/0.452 to 0.408/0.424 mm, residual jumps by 86.5%/86.5%, and dwell XY fluctuation by 36.8%/40.6%. The differently configured target device provides only a frozen transfer check: jumps and fluctuation fall by 60.0% and 28.9%, whereas centered XY rises from 0.478 to 0.484 mm; its complete results remain supplementary. These results support simulation-guided, real-supervised stability and localization refinement, not universal improvement across devices or legacy motions. All controls and negative results are retained. The surrogate is not a complete physical digital twin, and command-derived references are not independently measured contact truth.

**Keywords** Soft magnetic tactile sensing; dynamic simulation; causal history; motion-aware learning; trajectory reconstruction; reproducible evaluation

## I Introduction

Contact deformation of a soft magnetic layer changes the signals of fixed Hall nodes. Beyond current position, indentation, tangential motion, reversals, and prior dwell affect these signals: identical positions need not produce identical fields. Regular circles, radial motions, and normal indentation alone provide limited coverage of the joint position-history distribution. Coordinate jumps and dwell fluctuations impair both interpretation and trajectory-dependent control.

Simulation offers more than sample multiplication. A continuous forward model holds positions and sampling schedules fixed while varying arrival history, speed, or dwell, turning mixed effects into controlled questions. Linked displays of indenter motion, node vectors, magnetic channels, and independently computed inverse outputs distinguish stopped motion from equilibrated response. Such a platform can identify weak coverage and motivate testable representations; success inside it does not substitute for real validation.

Soft magnetic sensing has prior mechanical-magnetic design models [1] and material-point response simulation [2], while simulation-to-real tactile learning also has an optical-sensing precedent [3]. Different sensing mechanisms, materials, and geometries are not interchangeable accuracy guarantees. Our empirical state-space surrogate predicts five-node fields for a known indenter, not a unique volumetric deformation, arbitrary contact geometry, or full magnetic field. Virtual labels come from commanded simulation inputs, not inverse-network predictions recycled as truth.

Earlier virtual augmentation and history readouts showed that real stability benefits may coexist with localization bias. We ask whether simulation-motivated context can refine an already optimized system while real position and motion supervision protects localization, without added XYZ output smoothing. Contributions are an equilibrium-preserving axis-separated surrogate and runnable platform; function-preserving causal feature expansion retaining existing histories; a matched motion-aware objective comparison; and complete frozen-real evaluation with unseen complex motions, legacy capabilities, and unloaded controls. The refined candidates do not continue training virtual weights: simulation motivates the representation, whereas both candidates initialize from the real baseline and train on source-real data.

## II Dynamic Forward Surrogate

### A Coupling between position and deformation memory

Let q be the commanded indenter coordinate, s the latent deformation-history state, and S the five-node observation operator. Equation (1) summarizes the observation. B denotes the contact- and history-dependent spatial field, B0 a confirmed unloaded baseline, and noise includes sampling disturbances and unmodeled effects. Positive nominal Z denotes indentation; extremely shallow commanded motion is not direct ground truth for actual contact depth.

$$\Delta\mathbf b_t=\mathcal S\{\mathbf B(\mathbf r;q_t,s_t)\}-\mathbf b_0+\boldsymbol\epsilon_t,\qquad q_t=(X_t,Y_t,Z_t).\tag{1}$$

The same position can carry different history states and therefore different responses. The local first-order change has position and history contributions. A static inverse that explains the second term as displacement can produce direction-dependent errors, reversal tails, or incorrect branches. Causal history estimation separates these contributions; it does not predict future coordinates from a trajectory template.

$$\delta\mathbf b_t\simeq J_q\delta q_t+J_s\delta s_t+\boldsymbol\epsilon_t.\tag{2}$$

This is a conditionally observable inverse problem. Within a valid local branch, suppose admissible history uncertainty spans a response-space subspace H. After removing H, the position Jacobian must retain three independent directions. Equation (3) is a local first-order separation criterion. Practical resolution further requires the smallest singular value multiplied by the target displacement to exceed effective magnetic noise. We use this criterion to explain state constraints, not as a global uniqueness theorem for the entire workspace.

$$\operatorname{rank}(P_{H^\perp}J_q)=3,\qquad \sigma_{\min}(P_{H^\perp}J_q)>0.\tag{3}$$

H represents effective uncertainty under causal constraints. The 660 empirical history coordinates below cannot be interpreted as physical degrees of freedom independently observable from fifteen instantaneous channels. History constraints, the static magnetic atlas, and output permission jointly reduce the feasible solution set. Uncertain states must remain possible outside the supported workspace, near the noise level, or where position and history responses strongly overlap.

### B Nonlinear spatial encoding

Spatial features comprise ten low-order terms and one hundred three-dimensional Gaussian functions. With x=X/6, y=Y/6, and d=max(Z,0)/3, the polynomial terms are 1, x, y, d, x squared, xy, y squared, xd, yd, and d squared. Gaussian centers form a grid with X and Y at -6, -3, 0, 3, and 6 mm and Z at 0.6, 1.4, 2.2, and 3.0 mm.

$$r_j(q)=\exp\!\left[-\frac12\sum_{a\in\{X,Y,Z\}}\left(\frac{q_a-c_{j,a}}{\sigma_a}\right)^2\right],\qquad\boldsymbol\sigma=(2.5,2.5,0.65)\ {\rm mm}.\tag{4}$$

All 110 features are multiplied by an indentation factor. Static increments vanish for a nominal unloaded input, while historical residuals can persist. The 0.2 mm scale is an empirical feature parameter, not an identified contact threshold or material constant. Gaussian encoding describes smooth local spatial changes without assuming a linear XYZ-to-field relationship.

$$\phi(q)=\frac{Z^+}{Z^++0.2}\,[p(q),r_1(q),\ldots,r_{100}(q)]^{\mathsf T},\qquad Z^+=\max(Z,0).\tag{5}$$

### C Symmetric axis separation and static equilibrium

From previous and current positions, construct a=(X(t),Y(t),Z(t-1)) and c=(X(t-1),Y(t-1),Z(t)). Average the feature changes obtained by applying tangential motion first and normal motion first, as in (6). The two increments sum exactly to the total feature change. Pure Z motion has zero tangential increment, and pure XY motion has zero normal increment. For combined motion, cross-effects are shared symmetrically rather than assigned through an arbitrary update order.

$$\delta_{xy,t}=\frac12[\phi(a_t)-\phi(q_{t-1})+\phi(q_t)-\phi(c_t)],\qquad\delta_{z,t}=\phi(q_t)-\phi(q_{t-1})-\delta_{xy,t}.\tag{6}$$

Each axis-separated increment drives three exponentially decaying states with timescales of 0.3, 1.0, and 4.0 s. States use only present and past motion. A confirmed baseline reset, unknown coordinate, or gap longer than 0.5 s clears them rather than inventing an unknown preceding history.

$$h_{a,k,t}=\rho_{k,t}(h_{a,k,t-1}+\delta_{a,t}),\quad\rho_{k,t}=e^{-\Delta t/\tau_k},\quad a\in\{xy,z\}.\tag{7}$$

During dwell, feature increments vanish and history states decay toward a fixed static response. Reversal changes excitation signs, preserving distinct responses for different approaches to the same position. The static and history terms in (8) thus represent equilibrium and transients separately. Dynamic fitting cannot freely shift the frozen static base. These exponential modes are response-surrogate components, not three measured silicone relaxation constants.

$$\widehat{\Delta\mathbf b}_t=C_0^{\mathsf T}\phi(q_t)+\sum_{k=1}^{3}\left[C_{xy,k}^{\mathsf T}h_{xy,k,t}+C_{z,k}^{\mathsf T}h_{z,k,t}\right].\tag{8}$$

The model is linear in readout coefficients and nonlinear in motion. We freeze the 110-by-15 static coefficient matrix before fitting 660-by-15 historical coefficients. Features are scaled using source-training weighted second moments, and ridge regression constrains the fit. Each complete dynamic record receives equal total weight regardless of frame count; transition-boundary frame weights are halved. Timescales, axis separation, and ridge strength are selected only on source validation records.

$$\widehat C=\arg\min_C\sum_iw_i\|H_iC-(\Delta\mathbf b_i-C_0^{\mathsf T}\phi_i)\|_2^2+\lambda\|D_\phi C\|_F^2.\tag{9}$$

D-phi is the frozen training feature-scale matrix. The selected historical ridge strength is 0.01 and the feature-scale floor is 0.02. Source validation requires pure-Z and grid errors to stay within 110% of the static reference and retains a static fallback candidate.

### D How virtual data can change reconstruction

New motion sequences expand the joint distribution of position and history rather than merely repeat an existing circle. Curvature, speed, reversal, and dwell pair similar positions with different histories, providing constraints that help the inverse network distinguish spatial change from persistent response. Online inputs remain current and past magnetic observations, not the generated action coordinates.

Data quantity cannot remove generator bias. If D is locally Lipschitz within a fixed valid branch, the matched real and simulated inputs satisfy the conditional bound (10). Simulation error can be amplified by local inverse sensitivity. Low field NRMSE or similar mean reconstruction errors therefore do not establish pointwise reproduction.

$$\|D(\widetilde{\mathbf b}_{\le t})-q_t\|\le\|D(\mathbf b_{\le t})-q_t\|+L_D\|\widetilde{\mathbf b}_{\le t}-\mathbf b_{\le t}\|.\tag{10}$$

The bound applies only to a fixed-length causal window and the same valid output branch. Rejection and discrete branch changes require separate coverage and failure statistics. We consequently distinguish response prediction, mechanism reproduction, new-motion learning, and real transfer, rather than equating simulator-internal improvement with hardware accuracy.

## III Simulation Platform and Virtual Data

### A Physical data source and continuous interfaces

Inputs originate from complete motion and fifteen-channel magnetic streams recorded on a three-axis loading platform. Five MLX90393 nodes under the flexible magnetic layer use four corner positions and an offset center; they are read through a CH341 interface. Host software archives command events, unloaded baselines, and configuration snapshots. Device structure and operating photographs are available in the project's [hardware and system media](https://github.com/googo2025/MagTouch-RA-3d-trajectory-recon/tree/main/media). We call the magnetic-layer assembly used for forward fitting and real refinement the source device; a separately profiled assembly is the target device. Both records share the nominal five-node layout, MLX90393 acquisition settings, silicone-material version, and a rounded-flat indenter (5 mm body, 3 mm top flat, R1 edge). Their profile identifiers are 5×5-had-239 and 5×5-had-Yellow; recorded contact-reference Z values are 23.360001 and 23.059999 mm. Magnet specification, layer thickness, and gap are incompletely recorded, so device differences cannot be uniquely assigned to a particular material or geometry parameter. Two source batches repeat the same assembly; the target participates in no model fitting, normalization, or selection and provides only a bounded frozen transfer check. Record metadata define indenter, magnetic layer, and mounting conditions. Command-derived nominal motion is not encoder position. Appendix A specifies the source data converted into the continuously driven surrogate.

![Six functional panels of the simulation platform](../figures/simulation_platform_montage.png)

**Figure 1.** Six actual running-interface panels: (1) continuous 3D motion, indentation, and withdrawal; (2) five-node, fifteen-channel response; (3) independently generated free XYZ motion; (4) virtual streams passed to a frozen inverse and replay of its computed output; (5) four-node prediction of a known held-out node; and (6) paired conditions and inverse-module ablations. Panels retain the original interface language; numbering and captions identify their functions. Example interface values are not pooled into paper statistics. Node vectors are not field lines or finite-element displacements, and generated motion is not mislabeled as reconstructed motion.

The platform supports continuous manual XY movement, indentation adjustment, hold and withdrawal, automatic path replay, and provenance-bearing export. Three-axis readings at five nodes update with motion; paths and coordinate sequences can be inspected together. Complete Python inverse outputs are saved independently and displayed by the website, which does not substitute input paths for reconstruction. A finite-element viewer can receive external mesh and displacement files but did not generate this paper's data.

### B Experiment organization and motion expansion

![Forward modeling and training workflow](../figures/causal_simulation_pipeline.png)

**Figure 2.** Source data supports forward modeling and causal-representation exploration, followed by source-real H1/H2 training and frozen evaluation of all new motions and controls. The generator does not read inverse predictions; target and new evaluation data do not fit the candidates. Simulation motivation is not initialization from virtual weights.

Four constructions generate spatial Bezier curves, inertial random walks, free 3D waypoints, and multiscale curves. Every complete path has an independent geometry seed; paths are not copies of recorded motions. Nominal contact generation uses X and Y from -5 to 5 mm, Z from 0.45 to 2.4 mm, speed from 0.5 to 2 mm/s, dwell from 0 to 2.5 s, and 10 Hz sampling, with indentation and withdrawal segments retained. Source fitting covers approximately 0.40-2.30 mm indentation, so the upper generated edge is not a physically validated coverage guarantee.

Training expands predetermined prefixes of 100, 500, and 1000 paths. The independent virtual test has 64 complete paths with nonoverlapping geometry seeds. Source unloaded records estimate added channel noise. This scale study does not use domain randomization, and 1000 paths from one empirical generator are not 1000 independent devices.

## IV Simulation-Motivated Real-Supervised Reconstruction

### A Retaining existing history while expanding causal context

The fully trained baseline is not an instantaneous-field network. Its 53 inputs contain three guarded candidate coordinates, sixteen corrected magnetic features, a rank-two slow state, and sixteen differences at each of 0.1 and 0.4 s. A 53-64-64-2 tanh head predicts planar residuals; depth, state, and output permission form the complete causal chain. Temporal information is therefore retained, not introduced for the first time.

Let z(t) be the original sixteen-dimensional corrected features. Three EMAs and current-minus-EMA residuals use actual time intervals. The appended 48 features form a 101-64-64-2 network with 10818 parameters. Original 53-feature means and scales remain elementwise identical; only source TRAIN supplies the new means and scales, with scale floor 0.25. The network predicts correction r and the planar candidate is guarded base g plus r.

$$m_{k,t}=\rho_{k,t}m_{k,t-1}+(1-\rho_{k,t})z_t,\quad v_{k,t}=z_t-m_{k,t},\quad\rho_{k,t}=e^{-\Delta t/\tau_k}.\tag{11}$$

$$u_t=[u^{(0)}_t,v_{1,t},v_{2,t},v_{3,t}],\qquad\widehat q^{xy}_t=g_t+N_\theta(u_t).\tag{12}$$

Time scales are 0.3, 1.0, and 4.0 s, motivated by the surrogate rather than identified material constants. Resets follow original XY invalidity, B0/stream resets, non-increasing time, and gaps above 0.25 s. All original parameters are copied and 48 zero rows are added to the first-layer weights. Initial predictions match exactly on all 11415 training samples and on a 160-frame complete-chain hardware-fixture test. The latter is a software test, not hardware validation. H1 and H2 have equal additional capacity; comparing either with H0 is not a pure history-information ablation.

### B Matched objectives and real-supervision boundaries

We retain 32 whole source TRAIN Trials and three whole VAL Trials, without random frame splitting. The baseline uses 7217 dynamic samples and 565 static anchors. Candidates add 3633 source-TRAIN dwell samples, totaling 11415. Qualified dwell requires valid fields, finite nominal XYZ, logged EST segments, depth at least 0.45 mm, no motion/boundary, and dwell age 0.2-1.5 s; every fourth eligible supplemental row is retained. Discarded batches and other indenter domains are not mixed. Nominal extremely shallow coordinates are not contact truth.

Position residual targets are reference XY minus g, clipped to +/-3 mm per axis. Dynamic, supplemental-dwell, and static-anchor groups receive total loss masses 0.8, 0.1, and 0.1. H1 uses position loss and L2. H2 additionally matches predicted position increments to reference increments, adds the same increment penalty on qualified dwells, and protects zero static correction. Pairs are consecutive valid rows of the same Trial and logged EST segment, with 0<dt<=0.25 s and no command boundary; 8833 motion pairs and 3439 qualified dwell pairs receive their respective set-mean weighting.

$$\mathcal L_1=0.8\mathbb E_D\|r_\theta-y\|^2+0.1\mathbb E_H\|r_\theta-y\|^2+0.1\mathbb E_A\|r_\theta\|^2+\frac{0.001}{2}\sum_l\|W_l\|_F^2.\tag{13}$$

$$\mathcal L_2=\mathcal L_1+0.25\mathbb E_P\|\Delta\widehat q^{xy}-\Delta q^{xy}_{\rm ref}\|^2+0.25\mathbb E_{P_H}\|\Delta\widehat q^{xy}-\Delta q^{xy}_{\rm ref}\|^2+0.25\mathbb E_A\|r_\theta\|^2.\tag{14}$$

Each added coefficient is preset to 0.25; selection uses only source regular VAL. The increment loss matches reference movement rather than penalizing all changes toward zero, and dwell penalties are confined to qualified segments. H1/H2 share data, initialization, normalization, architecture, and optimizer, differing in explicit objectives. Only XY feature/head routes change in the complete chain. Z and permission parameters remain fixed, but coupling through candidate coordinates is evaluated.

### C Optimization completion, selection, and data roles

Adam uses seed 7, float64, batches of 256, beta1=0.9, beta2=0.999, and epsilon=10 to the power -8. L2 is 0.001/2 times the sum of squared weight matrices. Each epoch shuffles without replacement and traverses all 11415 samples in 45 updates, with a preset ceiling of 2000 epochs. Starting LR is 0.0002. Fifty epochs without relative 0.0001 source-VAL improvement halve it, down to 0.000001; another minimum-rate 50-epoch plateau stops training. Ceiling truncation is incomplete. Both candidates complete the plateau protocol.

The checkpoint objective is the three-Trial mean of raw XY error plus 0.1 residual-step RMS and 0.1 dwell RMS, plus 0.1 static-correction RMS. Admissibility additionally requires mean VAL raw XY no greater than baseline+0.005 mm, every Trial no greater than baseline+0.02 mm, and anchor RMS no greater than baseline+0.02 mm. The initial checkpoint remains eligible, avoiding forced deterioration; both selected epochs are nonzero. Protocol completion is not a claim of global nonconvex optimality.

| Model | Samples | Epochs | Best | Updates | Final LR |
|---|---|---|---|---|---|
| H1 | 11415 | 1015 | 146 | 45675 | 0.000001 |
| H2 | 11415 | 854 | 269 | 38430 | 0.000001 |

**Table I.** Actual training. H0 is the already optimized reference, not a repeated continuation control. Both new candidates undergo eight LR reductions and a minimum-rate 50-epoch plateau. Source VAL selects the best epoch, not the final epoch; complete logs, checkpoints, and normalization hashes are indexed.

### D Frozen-real replay and paired statistics

Primary testing comprises 36 complex real Trials from two source-device batches and one unadapted target device: six geometry families at 0.7/1.3 mm/s, twelve Trials per batch. Each batch also has a legacy circle, pure-Z motion, and before/after unloaded records, yielding 48 Trials. New motions do not enter candidate TRAIN, VAL, normalization, or checkpoint selection. Records were previously inspected for other research and are therefore retrospective frozen evaluation, not newly prospective blind testing. The repeated source batch is not a third device.

Complete causal replay preserves raw acquisition-group timestamps, fifteen-dimensional B0, control prefixes, and resets, without resampling, retrospective time alignment, or additional XYZ filtering. The reference reuses independently audited, SHA-frozen native outputs. Candidates freeze before reading evaluation inputs. An independent statistical implementation recomputes 144 model-by-Trial output streams; independent numerical reduction is not independent physical ground truth.

Accuracy and jumps use common-valid moving frames of all three models, while dwell fluctuation uses common-valid dwell frames. Rejection is not zero error; own coverage and unloaded outputs are separately reported. With e equal to prediction minus reference, XY/XYZ use mean Euclidean distance and Z uses MAE. Adjacent common-valid pairs do not cross rejected frames or gaps above 0.25 s; the jump threshold is 0.2 mm. One constant vector fitted per Trial supplies a translation diagnostic, not online calibration or modified raw output.

$$J=\frac{100}{|P|}\sum_{t\in P}\mathbf{1}\{\|e^{xy}_t-e^{xy}_{t-1}\|>0.2\ {\rm mm}\}.\tag{15}$$

$$b=\frac{1}{|I|}\sum_{t\in I}e^{xy}_t,\qquad E_{xy}^{\rm centered}=\frac{1}{|I|}\sum_{t\in I}\|e^{xy}_t-b\|.\tag{16}$$

Dwell RMS removes each contiguous block's output mean; blocks require at least three frames and one second. Block means are aggregated within Trial and then equally across twelve Trials. References derive from commands and EST, not encoders, optical contacts, or force measurements. Whole Trials, not high-rate frames, are the statistical units. Two speeds are clustered within six geometry families; 10000 paired resamples provide descriptive intervals and 64 sign assignments provide exact paired diagnostics. One seed and nonrandom chosen geometries limit population inference.


## V Experimental Results

### A Simulation fidelity boundaries and direction discovery

Matched source real/virtual reconstruction has XYZ errors of 0.286/0.257 mm, while corresponding outputs still differ by 0.348 mm on average; source magnetic NRMSE is approximately 25.10%. Simulation therefore motivates testable history representations rather than providing pointwise physical truth. An earlier, distinct 1000-virtual-path model reduced raw common-frame XY error on seven previously collected target contact records from 0.73968 to 0.62990 mm (14.84%). Its weights and scoring contract are separate from the real-refined candidates here. Other favorable and failed virtual candidates remain in the supplementary evidence.

### B Core result: real refinement balances localization and stability

| Batch | Model | XY mm | Centered XY mm | XYZ mm | Z MAE mm |
|---|---|---|---|---|---|
| Source batch 1 | H0 | 0.436 | 0.215 | 0.480 | 0.165 |
| Source batch 1 | H1 | 0.415 | 0.190 | 0.459 | 0.163 |
| Source batch 1 | H2 | 0.408 | 0.195 | 0.454 | 0.164 |
| Source batch 2 | H0 | 0.452 | 0.224 | 0.487 | 0.156 |
| Source batch 2 | H1 | 0.429 | 0.196 | 0.464 | 0.153 |
| Source batch 2 | H2 | 0.424 | 0.204 | 0.460 | 0.155 |

**Table II.** Trial-equal localization on all 24 new complex trajectories from two source-device batches, each with six geometries and two speeds. H0 is the fully optimized complete reference; H1 adds multiscale history with real-position supervision; H2 adds motion, qualified-dwell, and anchor objectives on matched data and histories. Raw XY/XYZ are unregistered; centered XY is a post hoc per-Trial translation diagnostic, not a deployable correction. Batch 2 is a repeat on the same device, not another independent device.

H2 reduces raw XY from 0.436/0.452 to 0.408/0.424 mm (6.36%/6.29%); raw XYZ improves by 5.44%/5.47%, and centered XY by 9.23%/9.30%. Raw XY improves on 22/24 source Trials and centered XY on 24/24. Six-geometry paired descriptive intervals for raw-XY differences are [-0.0414,-0.0143] and [-0.0412,-0.0182] mm; these are not cross-device population confidence intervals. H1 has better centered source XY than H2, but less stability benefit below; the best numbers from two models are not combined.

| Batch | Model | Jump rate | Dwell RMS mm | Own coverage | Rejected frames |
|---|---|---|---|---|---|
| Source batch 1 | H0 | 23.23% | 0.11195 | 99.895% | 11 |
| Source batch 1 | H1 | 7.77% | 0.08314 | 99.895% | 11 |
| Source batch 1 | H2 | 3.13% | 0.07075 | 99.895% | 11 |
| Source batch 2 | H0 | 24.56% | 0.11706 | 99.907% | 11 |
| Source batch 2 | H1 | 8.21% | 0.08343 | 99.890% | 13 |
| Source batch 2 | H2 | 3.31% | 0.06950 | 99.890% | 13 |

**Table III.** Residual jump rate (>0.2 mm change in XY residual across adjacent common-valid moving frames), common-dwell mean-centered XY RMS, each model's own movement coverage, and rejected frames. There are no all-zero XYZ outputs in eligible moving score domains; rejected frames are not scored as zero errors.

H2 reduces jump rate from 23.23%/24.56% to 3.13%/3.31% (86.52%/86.51%) and dwell XY RMS from 0.11195/0.11706 to 0.07075/0.06950 mm (36.80%/40.63%). Both metrics improve on all 24 source Trials. Own coverage remains 99.895% in batch 1 and changes from 99.907% to 99.890% in batch 2; the gain is not obtained by suppressing most outputs. Frozen new-motion native causal replay retains actual B0 and timestamps. This is retrospective evaluation of previously inspected records, not prospective blinded testing; command-derived references are not independent contact-position truth.

![Aggregate source results](../figures/source_real_metrics.png)

**Figure 3.** Trial-equal raw XY, jump rate, and dwell RMS across all twelve source Trials in each batch. Fixed H0/H1/H2 weights and all eligible trajectories are included. High-rate frames are not treated as independent experimental replicates.

For trajectory shape and error time courses, two distinct geometries at the same 1.3 mm/s speed were selected post hoc, with matched Trials from both repeated source batches. Both H2 raw XY and jumps improve over H0 in the corresponding four examples. These illustrations do not estimate the overall benefit; Tables II/III retain all 24 trajectories. Curves use native coordinates and timestamps, leaving missing frames disconnected without translation, filtering, or interpolation.

![Source trajectory example 4](../figures/source_example_source_multiscale_curve.png)

**Figure 4.** Source batch 1, multiscale curve, 1.3 mm/s. Left: command reference and native H0/H2 XY; right: unregistered XY error over time. Post hoc illustration, not a substitute for full-batch statistics.

![Source trajectory example 5](../figures/source_example_source_depth_reversing_spiral.png)

**Figure 5.** Source batch 1, depth-reversing spiral, 1.3 mm/s. Left: command reference and native H0/H2 XY; right: unregistered XY error over time. Post hoc illustration, not a substitute for full-batch statistics.

![Source trajectory example 6](../figures/source_example_source_repeat_multiscale_curve.png)

**Figure 6.** Source batch 2, multiscale curve, 1.3 mm/s. Left: command reference and native H0/H2 XY; right: unregistered XY error over time. Post hoc illustration, not a substitute for full-batch statistics.

![Source trajectory example 7](../figures/source_example_source_repeat_depth_reversing_spiral.png)

**Figure 7.** Source batch 2, depth-reversing spiral, 1.3 mm/s. Left: command reference and native H0/H2 XY; right: unregistered XY error over time. Post hoc illustration, not a substitute for full-batch statistics.

**Section B summary and bounds.** The same frozen model improves both raw localization and dynamic stability in two source batches without an added XYZ output smoother. Permission is audited separately: both unloaded controls yield zero output. Among 7581 safe-lift reference frames in source batch 1, H0/H1/H2 output without external lift hints on 9/10/9 frames, respectively; all 7704 corresponding batch-2 frames have zero output. We do not claim that false outputs disappear. Legacy source circles improve, but the pure-Z control shows local regression. Source XY projection gain is near one, arguing against uniform amplitude collapse; no events qualify under the prespecified sharp-reversal rule, so response delay is unestimable rather than zero. Complete legacy, permission, amplitude, training-curve, and per-Trial evidence is supplementary.

The target device uses a different magnetic-layer configuration with the same nominal indenter and five-node geometry. It is excluded from generator and candidate fitting and selection, and tests only bounded frozen transfer. On twelve complex target trajectories, H2 reduces jumps by 60.0% and dwell fluctuation by 28.9%; raw XY changes from 0.521 to 0.512 mm, but centered XY increases from 0.478 to 0.484 mm. This is not a clear claim of cross-device localization generalization. Full target comparisons and legacy costs remain supplementary.

## VI Discussion

### A How simulation yields a verifiable refinement direction

History representation is not equivalent to low-pass filtering XYZ outputs. Residual features retain current corrected magnetic observations while explicitly describing departures from several causal averages, allowing different corrections for different arrival histories. Virtual readouts motivated this context; real supervision then corrected its position relationship, and motion-aware losses constrained arbitrary attenuation of true movements. The evidence chain is surrogate-supported exploration, source-real training, and frozen-real new-motion evaluation, not pointwise physical replication or universally successful transfer of virtual weights.

Matched H1/H2 results show that explicit motion/dwell objectives further reduce jumps and fluctuation and improve raw mean localization with identical history features. H1 nevertheless achieves lower centered shape error on the source. These are distinct trade-offs: H1's best shape statistic and H2's best stability statistic cannot be combined into one model. The baseline already has history; added dimensions, dwell supervision, and capacity also change. The independent effect of every added time scale is not isolated.

### B Stability is not correct position

A constant bias leaves residual increments unchanged and disappears after dwell centering, yet raises raw localization error. The squared-error bias-scatter identity permits reduced fluctuation with increased bias; it does not numerically decompose our mean Euclidean distances. Conditional-mean regression [5] may average ambiguous movements, suppressing spikes while attenuating amplitude. The theoretical connection of noise training to derivative regularization [4] is a competing explanation, not a mechanism established by smooth curves.

$$\mathbb E\|e\|^2=\|\mathbb E e\|^2+\operatorname{tr}\operatorname{Cov}(e).\tag{17}$$

Both raw and centered source XY errors decrease here, so the improvement is not created merely by a fitted translation. Slight target centered degradation and legacy circle/pure-Z costs expose limited spatial/history coverage. Coordinate origins, fifteen-dimensional B0, gain, and history mismatch must be distinguished; subtracting a fitted output vector is not physical calibration. Jump-reduction and localization-increase percentages cannot be subtracted as a total benefit.

### C Constraints on mechanism and training-data interpretation

A separate frozen diagnostic on three source VAL Trials perturbs only the original sixteen TRAIN-standardized current corrected features, holding guards, slow state, and all history inputs fixed. Finite differences agree within 1.08 times 10 to the power -10. Trial-equal Frobenius means for H0/H1/H2 are 1.827/1.882/1.908 mm. This conditional current-feature gain does not decrease, ruling out a simple claim that the observed gain arises from a flatter current-feature head. It is not an end-to-end causal magnetic-noise transfer test and does not identify a unique mechanism.

Source XY projection gains remain near one, and H2 target gains do not shrink further, arguing against uniform amplitude attenuation alone. No events satisfy the sharp-reversal contract, so local lag remains unquantified; zero-delay claims are unsupported. Real datasets should cover position by history by speed while retaining static anchors, qualified dwells, and opposite movements, rather than simply multiplying similar paths. Motion loss should match reliable reference increments, not suppress all changes. Boundary and uncertain-arrival segments cannot be treated as dwell truth.

### D Evidence boundaries and reproducibility

The study uses one optimization seed, two devices, and six nonrandom geometry families. A repeated source batch supports repeatability, not an extra independent device. Descriptive intervals do not establish cross-device, cross-seed, or random-motion confidence. Previously inspected records are not relabeled prospective blind tests. Unknown indenters, multiple contacts, tilt, and rolling need corresponding geometry and joint-deformation validation; independent single-contact responses cannot simply be summed, nor can unfinished finite-element work substantiate them.

Supplementary evidence retains all virtual augmentation scales, targeted augmentation, denoising, action recognition, dwell hypotheses, and per-Trial negative outcomes. Production models, acquisition UI, and hardware remain unchanged. The runnable website is a static interactive frontend; GitHub Pages does not execute Python training, full causal replay, or finite-element solvers. Data, identifier mappings, SHA manifests, environment requirements, and reproduction entry points accompany an independent distribution. Unuploaded materials are not described as already public.

## VII Conclusion

Dynamic forward modeling, interactive simulation, causal-representation exploration, and source-real supervision form a reproducible research workflow. The main result is joint improvement of raw and centered localization, residual jumps, and dwell fluctuation on two source-device batches of complex motions excluded from fitting, without added XYZ output smoothing and while retaining the fully optimized complete baseline. The constrained-history candidate reduces jumps by about 86.5%, dwell fluctuation by 36.8%-40.6%, and raw XY error by about 6.3%. The target device also shows substantial stability benefits, with small localization changes and a slight centered cost.

These findings support simulation as a tool to discover useful representations and guide real-data refinement, not a requirement that every candidate or condition outperform the reference. Legacy-motion regressions, limited coverage changes, reference-measurement boundaries, and unquantified delay remain explicit. Simulation proposes directions; real data defines their domain. Stability and localization must be evaluated jointly rather than substituted for one another.


## References

[1] G. de Boer, N. Raske, H. Wang, J. Kow, A. Alazmani, M. Ghajari, P. Culmer, and R. Hewson. Design Optimisation of a Magnetic Field Based Soft Tactile Sensor. Sensors, 17, 2539, 2017. DOI 10.3390/s17112539.

[2] X. Lin, G. Lin, R. Liu, A. Wang, and Y. Lou. MPM-based simulation and bounded-error compression of material points for magnetic tactile sensors. Frontiers in Robotics and AI, 2026. DOI 10.3389/frobt.2026.1817251.

[3] C. Sferrazza and R. D'Andrea. Sim to real for high resolution optical tactile sensing From images to three dimensional contact force distributions. arXiv 2012.11295, version 3, 2021. https://arxiv.org/abs/2012.11295.

[4] C. M. Bishop. Training with Noise is Equivalent to Tikhonov Regularization. Neural Computation, 7(1), 108-116, 1995. https://www.microsoft.com/en-us/research/publication/training-with-noise-is-equivalent-to-tikhonov-regularization/.

[5] J. Lehtinen, J. Munkberg, J. Hasselgren, S. Laine, T. Karras, M. Aittala, and T. Aila. Noise2Noise: Learning Image Restoration without Clean Data. Proceedings of the 35th International Conference on Machine Learning, PMLR 80, 2965-2974, 2018. https://arxiv.org/abs/1803.04189.
