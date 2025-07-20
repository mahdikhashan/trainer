# Changelog

# [v2.0.0-rc.1](https://github.com/kubeflow/trainer/tree/v2.0.0-rc.1) (2025-07-03)

## New Features

- [release-2.0] feat: Add schedulingGates to PodSpecOverrides ([#2705](https://github.com/kubeflow/trainer/pull/2705) by [@astefanutti](https://github.com/astefanutti))
- [release-2.0] feat: Mutable PodSpecOverrides for suspended TrainJob ([#2698](https://github.com/kubeflow/trainer/pull/2698) by [@astefanutti](https://github.com/astefanutti))
- [Release 2.0] KEP-2170: Add the manifests overlay for Kubeflow Training V2 ([#2692](https://github.com/kubeflow/trainer/pull/2692) by [@Doris-xm](https://github.com/Doris-xm))

## Bug Fixes

- [release-2.0] fix(module): Change Go module name to v2 ([#2708](https://github.com/kubeflow/trainer/pull/2708) by [@andreyvelich](https://github.com/andreyvelich))
- [cherry-pick] fix(manifests): Update manifests to enable LLM fine-tuning workflow w… ([#2696](https://github.com/kubeflow/trainer/pull/2696) by [@Electronic-Waste](https://github.com/Electronic-Waste))
- [release-2.0] fix(plugins): Fix some errors in torchtune mutation process. ([#2693](https://github.com/kubeflow/trainer/pull/2693) by [@Electronic-Waste](https://github.com/Electronic-Waste))
- [release-2.0] fix(rbac): Add required RBAC to update ClusterTrainingRuntimes on OpenShift ([#2684](https://github.com/kubeflow/trainer/pull/2684) by [@astefanutti](https://github.com/astefanutti))

## Misc

- [release-2.0] chore: Copy generated CRDs into Helm charts ([#2704](https://github.com/kubeflow/trainer/pull/2704) by [@astefanutti](https://github.com/astefanutti))
- [cherry-pick] feat(example): Add alpaca-trianjob-yaml.ipynb. (#2670) ([#2702](https://github.com/kubeflow/trainer/pull/2702) by [@Electronic-Waste](https://github.com/Electronic-Waste))
- [release-2.0] chore: Replace the deprecated intstr.FromInt with intstr.FromInt32 ([#2697](https://github.com/kubeflow/trainer/pull/2697) by [@tenzen-y](https://github.com/tenzen-y))
- [release-2.0] chore: Remove the vendor specific parameters ([#2694](https://github.com/kubeflow/trainer/pull/2694) by [@tenzen-y](https://github.com/tenzen-y))
- [release-2.0] chore(runtime): Bump Torch to 2.7.1 and DeepSpeed to 0.17.1 ([#2687](https://github.com/kubeflow/trainer/pull/2687) by [@andreyvelich](https://github.com/andreyvelich))
- [release-2.0] chore(helm): Sync ClusterRule in Helm chart ([#2688](https://github.com/kubeflow/trainer/pull/2688) by [@astefanutti](https://github.com/astefanutti))

[Full Changelog](https://github.com/kubeflow/trainer/compare/v2.0.0-rc.0...v2.0.0-rc.1)

# [v2.0.0-rc.0](https://github.com/kubeflow/trainer/tree/v2.0.0-rc.0) (2025-06-10)

## Breaking Changes

- KEP-2170: Change API Group Name to `trainer.kubeflow.org` ([#2413](https://github.com/kubeflow/trainer/pull/2413) by [@Electronic-Waste](https://github.com/Electronic-Waste))
- Move generated Python models into kubeflow_trainer_api package ([#2632](https://github.com/kubeflow/trainer/pull/2632) by [@kramaranya](https://github.com/kramaranya))
- Upgrade kubernetes Go module version to 1.32 ([#2450](https://github.com/kubeflow/trainer/pull/2450) by [@tenzen-y](https://github.com/tenzen-y))
- Remove kubeflow-trainer prefix from jobset resource names ([#2596](https://github.com/kubeflow/trainer/pull/2596) by [@ChenYi015](https://github.com/ChenYi015))
- Remove the Training Operator V1 Source Code ([#2389](https://github.com/kubeflow/trainer/pull/2389) by [@andreyvelich](https://github.com/andreyvelich))

## New Features

### LLM Trainer V2

- KEP-2401: Support loading local LLMs ([#2644](https://github.com/kubeflow/trainer/pull/2644) by [@Electronic-Waste](https://github.com/Electronic-Waste))
- KEP-2401: Support mutating dataset preprocessing config in SDK ([#2638](https://github.com/kubeflow/trainer/pull/2638) by [@Electronic-Waste](https://github.com/Electronic-Waste))
- KEP-2401: Create LLM Training Runtimes for Llama 3.2 model family ([#2590](https://github.com/kubeflow/trainer/pull/2590) by [@Electronic-Waste](https://github.com/Electronic-Waste))
- KEP-2401: Complement torch plugin to support torchtune config mutation ([#2587](https://github.com/kubeflow/trainer/pull/2587) by [@Electronic-Waste](https://github.com/Electronic-Waste))
- KEP-2401: Create `torchtune` trainer image ([#2516](https://github.com/kubeflow/trainer/pull/2516) by [@Electronic-Waste](https://github.com/Electronic-Waste))
- KEP-2401: Refactor current `train()` API ([#2513](https://github.com/kubeflow/trainer/pull/2513) by [@Electronic-Waste](https://github.com/Electronic-Waste))
- KEP-2401: Kubeflow LLM Trainer V2 ([#2410](https://github.com/kubeflow/trainer/pull/2410) by [@Electronic-Waste](https://github.com/Electronic-Waste))

### Runtime Framework

- feat(runtimes): Support MLX Distributed Runtime with OpenMPI ([#2565](https://github.com/kubeflow/trainer/pull/2565) by [@andreyvelich](https://github.com/andreyvelich))
- feat(runtimes): Support DeepSpeed Runtime with OpenMPI ([#2559](https://github.com/kubeflow/trainer/pull/2559) by [@andreyvelich](https://github.com/andreyvelich))
- feat(runtime): remove needless Launcher chainer. ([#2558](https://github.com/kubeflow/trainer/pull/2558) by [@IRONICBo](https://github.com/IRONICBo))
- Store the TrainingRuntime numNodes as runtime.Info.PodSet.Count ([#2539](https://github.com/kubeflow/trainer/pull/2539) by [@tenzen-y](https://github.com/tenzen-y))
- Add dependencies to RuntimeRegistrar ([#2476](https://github.com/kubeflow/trainer/pull/2476) by [@tenzen-y](https://github.com/tenzen-y))
- KEP: 2170: Adding cel validations on TrainingRuntime/ClusterTrainingRuntime CRDs ([#2313](https://github.com/kubeflow/trainer/pull/2313) by [@akshaychitneni](https://github.com/akshaychitneni))
- Implement trainer.kubeflow.org/resource-in-use finalizer mechanism to ClusterTrainingRuntime ([#2625](https://github.com/kubeflow/trainer/pull/2625) by [@tenzen-y](https://github.com/tenzen-y))
- Implement trainer.kubeflow.org/resource-in-use finalizer mechanism to TrainingRuntime ([#2608](https://github.com/kubeflow/trainer/pull/2608) by [@tenzen-y](https://github.com/tenzen-y))

### MPI Plugin

- [feature]:add validations for MPIRuntime with RunLauncherAsNode ([#2551](https://github.com/kubeflow/trainer/pull/2551) by [@Harshal292004](https://github.com/Harshal292004))
- Implement CustomValidation UT for MPI plugin ([#2555](https://github.com/kubeflow/trainer/pull/2555) by [@tenzen-y](https://github.com/tenzen-y))
- Implemenet MPI Plugin for OpenMPI ([#2493](https://github.com/kubeflow/trainer/pull/2493) by [@tenzen-y](https://github.com/tenzen-y))
- Implement MPI plugin UTs ([#2481](https://github.com/kubeflow/trainer/pull/2481) by [@tenzen-y](https://github.com/tenzen-y))
- Implement MPIImplementation Enum CRD validation ([#2482](https://github.com/kubeflow/trainer/pull/2482) by [@tenzen-y](https://github.com/tenzen-y))
- Implement MPI numProcPerNode defaulter ([#2483](https://github.com/kubeflow/trainer/pull/2483) by [@tenzen-y](https://github.com/tenzen-y))
- Add MPIMLPolicySource CRD defaulters ([#2474](https://github.com/kubeflow/trainer/pull/2474) by [@tenzen-y](https://github.com/tenzen-y))
- Make MPIMLPolicySource optional fields as a pointer ([#2472](https://github.com/kubeflow/trainer/pull/2472) by [@tenzen-y](https://github.com/tenzen-y))
- KEP-2170: Implement MPI Plugin for Kubeflow Trainer ([#2394](https://github.com/kubeflow/trainer/pull/2394) by [@andreyvelich](https://github.com/andreyvelich))

### JobSet

- Retrieve JobSetSpec from runtime.Info in CustomValidations ([#2557](https://github.com/kubeflow/trainer/pull/2557) by [@tenzen-y](https://github.com/tenzen-y))
- KEP-2170: Deploy JobSet in `kubeflow-system` namespace ([#2388](https://github.com/kubeflow/trainer/pull/2388) by [@andreyvelich](https://github.com/andreyvelich))
- Bump JobSet to v0.8.0 ([#2463](https://github.com/kubeflow/trainer/pull/2463) by [@andreyvelich](https://github.com/andreyvelich))
- Upgrade jobset SDK version to v0.7.3 ([#2445](https://github.com/kubeflow/trainer/pull/2445) by [@Electronic-Waste](https://github.com/Electronic-Waste))

### New Examples

- Add question-answer example for v2 trainer ([#2580](https://github.com/kubeflow/trainer/pull/2580) by [@solanyn](https://github.com/solanyn))
- KEP-2170: Add PyTorch DDP MNIST training example ([#2387](https://github.com/kubeflow/trainer/pull/2387) by [@astefanutti](https://github.com/astefanutti))

### SDK Updates

- Remove SDK ([#2657](https://github.com/kubeflow/trainer/pull/2657) by [@eoinfennessy](https://github.com/eoinfennessy))
- feat(sdk): Get namespace from the provided context ([#2593](https://github.com/kubeflow/trainer/pull/2593) by [@andreyvelich](https://github.com/andreyvelich))
- feat(sdk): Support MPI-based TrainJobs ([#2545](https://github.com/kubeflow/trainer/pull/2545) by [@andreyvelich](https://github.com/andreyvelich))
- feat(sdk): Migrate to OpenAPI V3 ([#2490](https://github.com/kubeflow/trainer/pull/2490) by [@andreyvelich](https://github.com/andreyvelich))
- feat(sdk): Generate external Kubernetes and JobSet models ([#2466](https://github.com/kubeflow/trainer/pull/2466) by [@andreyvelich](https://github.com/andreyvelich))

## Bug Fixes

- Revert "fix(sdk): Fix type annotation for `train` method's `trainer` parameter" ([#2651](https://github.com/kubeflow/trainer/pull/2651) by [@Electronic-Waste](https://github.com/Electronic-Waste))
- fix(sdk): Fix bad arg passed to `get_args_using_torchtune_config` ([#2647](https://github.com/kubeflow/trainer/pull/2647) by [@eoinfennessy](https://github.com/eoinfennessy))
- fix(sdk): Fix type annotation for `train` method's `trainer` parameter ([#2646](https://github.com/kubeflow/trainer/pull/2646) by [@eoinfennessy](https://github.com/eoinfennessy))
- fix(controller): Fix RBAC permissions for TrainJob controller ([#2626](https://github.com/kubeflow/trainer/pull/2626) by [@andreyvelich](https://github.com/andreyvelich))
- Fix close-pr message in Stale GitHub Action ([#2622](https://github.com/kubeflow/trainer/pull/2622) by [@kramaranya](https://github.com/kramaranya))
- fix: remove redundant K8s version matrix from integration tests ([#2617](https://github.com/kubeflow/trainer/pull/2617) by [@tr33k](https://github.com/tr33k))
- fix(doc): tidy up KEP-2401. ([#2594](https://github.com/kubeflow/trainer/pull/2594) by [@Electronic-Waste](https://github.com/Electronic-Waste))
- Fix MPI Test runnable errors ([#2570](https://github.com/kubeflow/trainer/pull/2570) by [@tenzen-y](https://github.com/tenzen-y))
- Fix issue with fetching clustertrainingruntime for validations ([#2564](https://github.com/kubeflow/trainer/pull/2564) by [@akshaychitneni](https://github.com/akshaychitneni))
- fix(sdk): Add missing import types. ([#2566](https://github.com/kubeflow/trainer/pull/2566) by [@Electronic-Waste](https://github.com/Electronic-Waste))
- fix(sdk): Using correct entrypoint for mpirun ([#2552](https://github.com/kubeflow/trainer/pull/2552) by [@andreyvelich](https://github.com/andreyvelich))
- fix(sdk): add missing import type Initializer. ([#2541](https://github.com/kubeflow/trainer/pull/2541) by [@Electronic-Waste](https://github.com/Electronic-Waste))
- fix(ci): update `test-go` coverage ci config and replace trainer badge with new address. ([#2534](https://github.com/kubeflow/trainer/pull/2534) by [@IRONICBo](https://github.com/IRONICBo))
- fix(doc): Update `train()` API in KEP-2401 ([#2536](https://github.com/kubeflow/trainer/pull/2536) by [@Electronic-Waste](https://github.com/Electronic-Waste))
- fix(test): Update images for DockerHub publish ([#2535](https://github.com/kubeflow/trainer/pull/2535) by [@andreyvelich](https://github.com/andreyvelich))
- [hotfix] fix checkout on workflow ([#2531](https://github.com/kubeflow/trainer/pull/2531) by [@mahdikhashan](https://github.com/mahdikhashan))
- [hotfix] fix docker cred ([#2530](https://github.com/kubeflow/trainer/pull/2530) by [@mahdikhashan](https://github.com/mahdikhashan))
- fix: remove unused parameter name in default case of shouldUseCPU function ([#2521](https://github.com/kubeflow/trainer/pull/2521) by [@Diasker](https://github.com/Diasker))
- Fix #2407: Cap nproc_per_node based on CPU resources for PyTorch TrainJob ([#2492](https://github.com/kubeflow/trainer/pull/2492) by [@Diasker](https://github.com/Diasker))
- fix type in model initializer entrypoint ([#2489](https://github.com/kubeflow/trainer/pull/2489) by [@szaher](https://github.com/szaher))
- fix(runtime): fix error label name. ([#2487](https://github.com/kubeflow/trainer/pull/2487) by [@Electronic-Waste](https://github.com/Electronic-Waste))
- fix(sdk): resolve errors in deserialization ([#2457](https://github.com/kubeflow/trainer/pull/2457) by [@Electronic-Waste](https://github.com/Electronic-Waste))
- Fix missing external types in apply configurations ([#2429](https://github.com/kubeflow/trainer/pull/2429) by [@astefanutti](https://github.com/astefanutti))
- Fix API Group for Torch Runtime ([#2424](https://github.com/kubeflow/trainer/pull/2424) by [@andreyvelich](https://github.com/andreyvelich))
- Fix Kustomize patchesStrategicMerge deprecation warning ([#2405](https://github.com/kubeflow/trainer/pull/2405) by [@astefanutti](https://github.com/astefanutti))
- ControlPlane: Fix flaky integraion testings due to missing the latest version of object ([#2414](https://github.com/kubeflow/trainer/pull/2414) by [@tenzen-y](https://github.com/tenzen-y))

## Misc

- Tag Docker images with GitHub release tags ([#2662](https://github.com/kubeflow/trainer/pull/2662) by [@kramaranya](https://github.com/kramaranya))
- feat(controller): Implement PodSpecOverride API ([#2614](https://github.com/kubeflow/trainer/pull/2614) by [@andreyvelich](https://github.com/andreyvelich))
- Nominate @Electronic-Waste as approver and @astefanutti as reviewer ([#2659](https://github.com/kubeflow/trainer/pull/2659) by [@andreyvelich](https://github.com/andreyvelich))
- chore(build): Support Podman to run OpenAPI generator ([#2656](https://github.com/kubeflow/trainer/pull/2656) by [@astefanutti](https://github.com/astefanutti))
- chore(docs): Add OpenSSF Best Practices Badge ([#2611](https://github.com/kubeflow/trainer/pull/2611) by [@andreyvelich](https://github.com/andreyvelich))
- [chore] update stale action version to latest ([#2642](https://github.com/kubeflow/trainer/pull/2642) by [@mahdikhashan](https://github.com/mahdikhashan))
- Remove TrainJobCreated condition ([#2621](https://github.com/kubeflow/trainer/pull/2621) by [@astefanutti](https://github.com/astefanutti))
- ci: refactor build-push-images workflow ([#2607](https://github.com/kubeflow/trainer/pull/2607) by [@milinddethe15](https://github.com/milinddethe15))
- Update Go to v1.24 (#2615) ([#2620](https://github.com/kubeflow/trainer/pull/2620) by [@vzamboulingame](https://github.com/vzamboulingame))
- test(runtime): add UT for IndexTrainJobTrainingRuntime ([#2603](https://github.com/kubeflow/trainer/pull/2603) by [@Harshal292004](https://github.com/Harshal292004))
- ci: add k8s `v1.32` for tests env ([#2613](https://github.com/kubeflow/trainer/pull/2613) by [@milinddethe15](https://github.com/milinddethe15))
- chore(deps): bump torch from 2.5.0 to 2.6.0 in /cmd/runtimes/deepspeed ([#2606](https://github.com/kubeflow/trainer/pull/2606) by [@dependabot[bot]](https://github.com/apps/dependabot))
- chore(deps): bump golang.org/x/net from 0.36.0 to 0.38.0 ([#2602](https://github.com/kubeflow/trainer/pull/2602) by [@dependabot[bot]](https://github.com/apps/dependabot))
- test(runtime): add UT for jobset runtime valid function. ([#2562](https://github.com/kubeflow/trainer/pull/2562) by [@Harshal292004](https://github.com/Harshal292004))
- Add Helm chart for kubeflow trainer ([#2435](https://github.com/kubeflow/trainer/pull/2435) by [@ChenYi015](https://github.com/ChenYi015))
- chore(test): Removed the no longer needed github-trigger-rerun-test.yaml ([#2589](https://github.com/kubeflow/trainer/pull/2589) by [@hbelmiro](https://github.com/hbelmiro))
- Add PodNetwork plugin to KEP-2170 Job Pipeline Framework description ([#2578](https://github.com/kubeflow/trainer/pull/2578) by [@tenzen-y](https://github.com/tenzen-y))
- chore(docs): Update Slack channel ([#2569](https://github.com/kubeflow/trainer/pull/2569) by [@andreyvelich](https://github.com/andreyvelich))
- docs: update CONTRIBUTING.md for Kubeflow Trainer V2 ([#2561](https://github.com/kubeflow/trainer/pull/2561) by [@muzzlol](https://github.com/muzzlol))
- test(runtime): add UT for torch runtime valid function. ([#2560](https://github.com/kubeflow/trainer/pull/2560) by [@IRONICBo](https://github.com/IRONICBo))
- feat(doc): add Runtime API design in KEP-2401. ([#2501](https://github.com/kubeflow/trainer/pull/2501) by [@Electronic-Waste](https://github.com/Electronic-Waste))
- Update CONTRIBUTING.md ([#2512](https://github.com/kubeflow/trainer/pull/2512) by [@MuhammedgitAli](https://github.com/MuhammedgitAli))
- feat: add replicatedJobs.replicas validations in validateReplicatedJobs function. ([#2533](https://github.com/kubeflow/trainer/pull/2533) by [@IRONICBo](https://github.com/IRONICBo))
- Construct Trainer based on trainer.kubeflow.org/trainjob-ancestor-step label ([#2548](https://github.com/kubeflow/trainer/pull/2548) by [@tenzen-y](https://github.com/tenzen-y))
- chore: Enable GCI for golangci-lint ([#2540](https://github.com/kubeflow/trainer/pull/2540) by [@tenzen-y](https://github.com/tenzen-y))
- [feature] merge GHCR and DockerHub CI jobs ([#2537](https://github.com/kubeflow/trainer/pull/2537) by [@ashwinr64](https://github.com/ashwinr64))
- feat(controller): Refactor the Initializer APIs of TrainJob ([#2523](https://github.com/kubeflow/trainer/pull/2523) by [@andreyvelich](https://github.com/andreyvelich))
- Migrate InfoOptions.podSpecReplias and info.Scheduler.TotalRequests to info.TemplateSpec.PodSet ([#2524](https://github.com/kubeflow/trainer/pull/2524) by [@tenzen-y](https://github.com/tenzen-y))
- [feature] pull images in manifest from ghcr ([#2529](https://github.com/kubeflow/trainer/pull/2529) by [@mahdikhashan](https://github.com/mahdikhashan))
- [feature] migrate images to ghcr ([#2455](https://github.com/kubeflow/trainer/pull/2455) by [@mahdikhashan](https://github.com/mahdikhashan))
- KEP-2170: Adding validation webhook for v2 trainjob ([#2307](https://github.com/kubeflow/trainer/pull/2307) by [@akshaychitneni](https://github.com/akshaychitneni))
- Migrate Info.Trainer to Info.TemplateSpec.PodSet ([#2520](https://github.com/kubeflow/trainer/pull/2520) by [@tenzen-y](https://github.com/tenzen-y))
- Implement E2E for OpenMPI workload ([#2500](https://github.com/kubeflow/trainer/pull/2500) by [@tenzen-y](https://github.com/tenzen-y))
- Bump golang.org/x/net from 0.33.0 to 0.36.0 ([#2514](https://github.com/kubeflow/trainer/pull/2514) by [@dependabot[bot]](https://github.com/apps/dependabot))
- Move TrainJob marker defaulting and validation integration tests to test/integration/webhooks pkg ([#2486](https://github.com/kubeflow/trainer/pull/2486) by [@tenzen-y](https://github.com/tenzen-y))
- feat(controller): Integrate DependsOn API ([#2484](https://github.com/kubeflow/trainer/pull/2484) by [@andreyvelich](https://github.com/andreyvelich))
- Store E2E manifests to artifacts directory ([#2478](https://github.com/kubeflow/trainer/pull/2478) by [@tenzen-y](https://github.com/tenzen-y))
- Use large runner for building container image ([#2475](https://github.com/kubeflow/trainer/pull/2475) by [@tenzen-y](https://github.com/tenzen-y))
- chore(test): Upload artifacts from dir ([#2473](https://github.com/kubeflow/trainer/pull/2473) by [@andreyvelich](https://github.com/andreyvelich))
- Implement UTs for PlainML plugin ([#2469](https://github.com/kubeflow/trainer/pull/2469) by [@tenzen-y](https://github.com/tenzen-y))
- chore(test): Add E2E tests for Kubeflow Trainer ([#2470](https://github.com/kubeflow/trainer/pull/2470) by [@andreyvelich](https://github.com/andreyvelich))
- KEP-2170: Add Kubeflow Trainer Pipeline Framework Design ([#2439](https://github.com/kubeflow/trainer/pull/2439) by [@tenzen-y](https://github.com/tenzen-y))
- Replace Kueue PodRequests helper with core k/k one ([#2461](https://github.com/kubeflow/trainer/pull/2461) by [@tenzen-y](https://github.com/tenzen-y))
- KEP-2170: Use SSA to reconcile TrainJob components ([#2431](https://github.com/kubeflow/trainer/pull/2431) by [@astefanutti](https://github.com/astefanutti))
- Bump golang.org/x/net from 0.30.0 to 0.33.0 ([#2451](https://github.com/kubeflow/trainer/pull/2451) by [@dependabot[bot]](https://github.com/apps/dependabot))
- Use the correct apiVersion name ([#2444](https://github.com/kubeflow/trainer/pull/2444) by [@runzhen](https://github.com/runzhen))
- Add 'KEP Usage' KEP and template link ([#2423](https://github.com/kubeflow/trainer/pull/2423) by [@anishasthana](https://github.com/anishasthana))
- KEP-2170: Add validation to Torch `numProcPerNode` field ([#2409](https://github.com/kubeflow/trainer/pull/2409) by [@astefanutti](https://github.com/astefanutti))
- update migration url on readme file ([#2436](https://github.com/kubeflow/trainer/pull/2436) by [@varodrig](https://github.com/varodrig))
- IntegraionTests: Waiting for expected conditions before emulate JobSet controller manager ([#2425](https://github.com/kubeflow/trainer/pull/2425) by [@tenzen-y](https://github.com/tenzen-y))
- Nominate @Electronic-Waste as a reviewer ([#2427](https://github.com/kubeflow/trainer/pull/2427) by [@andreyvelich](https://github.com/andreyvelich))
- Update the naming conventions for Kubeflow Trainer ([#2415](https://github.com/kubeflow/trainer/pull/2415) by [@andreyvelich](https://github.com/andreyvelich))
- Rename paddlepaddle_defaults.go file name ([#2399](https://github.com/kubeflow/trainer/pull/2399) by [@ChristianZaccaria](https://github.com/ChristianZaccaria))
- Bump golang.org/x/net from 0.30.0 to 0.33.0 ([#2391](https://github.com/kubeflow/trainer/pull/2391) by [@dependabot[bot]](https://github.com/apps/dependabot))
- KEP-2170: Add unit and Integration tests for model and dataset initializers ([#2323](https://github.com/kubeflow/trainer/pull/2323) by [@seanlaii](https://github.com/seanlaii))
- Testing CI in JAX example ([#2385](https://github.com/kubeflow/trainer/pull/2385) by [@saileshd1402](https://github.com/saileshd1402))
- Upgrade huggingface_hub to v0.27.x in dataset initializer v2 ([#2379](https://github.com/kubeflow/trainer/pull/2379) by [@astefanutti](https://github.com/astefanutti))
- Add Changelog for Training Operator v1.9.0-rc.0 ([#2380](https://github.com/kubeflow/trainer/pull/2380) by [@andreyvelich](https://github.com/andreyvelich))
- Add release branch to the image push trigger ([#2376](https://github.com/kubeflow/trainer/pull/2376) by [@andreyvelich](https://github.com/andreyvelich))

[Full Changelog](https://github.com/kubeflow/trainer/compare/v1.8.1...v2.0.0-rc.0)

# [v1.9.0](https://github.com/kubeflow/training-operator/tree/v1.9.0) (2025-01-21)

## Breaking Changes

- Upgrade Kubernetes to v1.31.3 ([#2330](https://github.com/kubeflow/training-operator/pull/2330) by [@astefanutti](https://github.com/astefanutti))
- Upgrade Kubernetes to v1.30.7 ([#2332](https://github.com/kubeflow/training-operator/pull/2332) by [@astefanutti](https://github.com/astefanutti))
- Update the name of PVC in `train` API ([#2187](https://github.com/kubeflow/training-operator/pull/2187) by [@helenxie-bit](https://github.com/helenxie-bit))
- Remove support for MXJob ([#2150](https://github.com/kubeflow/training-operator/pull/2150) by [@tariq-hasan](https://github.com/tariq-hasan))
- Support Python 3.11 and Drop Python 3.7 ([#2105](https://github.com/kubeflow/training-operator/pull/2105) by [@tenzen-y](https://github.com/tenzen-y))

## New Features

### Distributed JAX

- Add JAX controller ([#2194](https://github.com/kubeflow/training-operator/pull/2194) by [@sandipanpanda](https://github.com/sandipanpanda))
- Add JAX API ([#2163](https://github.com/kubeflow/training-operator/pull/2163) by [@sandipanpanda](https://github.com/sandipanpanda))
- JAX Integration Enhancement Proposal ([#2125](https://github.com/kubeflow/training-operator/pull/2125) by [@sandipanpanda](https://github.com/sandipanpanda))
- JAX example for MNIST SPMD and add CI testing ([#2390](https://github.com/kubeflow/training-operator/pull/2390) by [@saileshd1402](https://github.com/saileshd1402))

### New Examples

- FSDP Example for T5 Fine-Tuning and PyTorchJob ([#2286](https://github.com/kubeflow/training-operator/pull/2286) by [@andreyvelich](https://github.com/andreyvelich))
- Add DeepSpeed Example with Pytorch Operator ([#2235](https://github.com/kubeflow/training-operator/pull/2235) by [@Syulin7](https://github.com/Syulin7))

### Control Plane Updates

- Validate pytorchjob workers are configured when elasticpolicy is configured ([#2320](https://github.com/kubeflow/training-operator/pull/2320) by [@tarat44](https://github.com/tarat44))
- [Feature] Support managed by external controller ([#2203](https://github.com/kubeflow/training-operator/pull/2203) by [@mszadkow](https://github.com/mszadkow))
- Update trainer to ensure type consistency for `train_args` and `lora_config` ([#2181](https://github.com/kubeflow/training-operator/pull/2181) by [@helenxie-bit](https://github.com/helenxie-bit))
- Support ARM64 platform in TensorFlow examples ([#2119](https://github.com/kubeflow/training-operator/pull/2119) by [@akhilsaivenkata](https://github.com/akhilsaivenkata))
- Feat: Support ARM64 platform in XGBoost examples ([#2114](https://github.com/kubeflow/training-operator/pull/2114) by [@tico88612](https://github.com/tico88612))
- ARM64 supported in PyTorch examples ([#2116](https://github.com/kubeflow/training-operator/pull/2116) by [@danielsuh05](https://github.com/danielsuh05))

### SDK Updates

- [SDK] Adding env vars ([#2285](https://github.com/kubeflow/training-operator/pull/2285) by [@tarekabouzeid](https://github.com/tarekabouzeid))
- [SDK] Use torchrun to create PyTorchJob from function ([#2276](https://github.com/kubeflow/training-operator/pull/2276) by [@andreyvelich](https://github.com/andreyvelich))
- [SDK] move env var to constants.py ([#2268](https://github.com/kubeflow/training-operator/pull/2268) by [@varshaprasad96](https://github.com/varshaprasad96))
- [SDK] Allow customising base trainer and storage images in Train API ([#2261](https://github.com/kubeflow/training-operator/pull/2261) by [@varshaprasad96](https://github.com/varshaprasad96))
- [SDK] Read namespace from the current context ([#2255](https://github.com/kubeflow/training-operator/pull/2255) by [@andreyvelich](https://github.com/andreyvelich))
- [SDK] Sync Transformers version for train API ([#2146](https://github.com/kubeflow/training-operator/pull/2146) by [@andreyvelich](https://github.com/andreyvelich))
- [SDK] Explain Python version support cycle ([#2144](https://github.com/kubeflow/training-operator/pull/2144) by [@andreyvelich](https://github.com/andreyvelich))

### Kubeflow Training V2

- KEP-2170: Kubeflow Training V2 API ([#2171](https://github.com/kubeflow/training-operator/pull/2171) by [@andreyvelich](https://github.com/andreyvelich))
- KEP-2170: Update V2 KEP with MPI Runtime info ([#2345](https://github.com/kubeflow/training-operator/pull/2345) by [@andreyvelich](https://github.com/andreyvelich))
- Always update TrainJob status on errors ([#2352](https://github.com/kubeflow/training-operator/pull/2352) by [@astefanutti](https://github.com/astefanutti))
- Fix TrainJob status comparison and update ([#2353](https://github.com/kubeflow/training-operator/pull/2353) by [@astefanutti](https://github.com/astefanutti))
- Add required RBAC on TrainJob finalizer sub-resources ([#2350](https://github.com/kubeflow/training-operator/pull/2350) by [@astefanutti](https://github.com/astefanutti))
- KEP-2170: [SDK] Initial implementation of the Kubeflow Training V2 Python SDK ([#2324](https://github.com/kubeflow/training-operator/pull/2324) by [@andreyvelich](https://github.com/andreyvelich))
- KEP-2170: Add Torch Distributed Runtime ([#2328](https://github.com/kubeflow/training-operator/pull/2328) by [@andreyvelich](https://github.com/andreyvelich))
- KEP-2170: Add TrainJob conditions ([#2322](https://github.com/kubeflow/training-operator/pull/2322) by [@tenzen-y](https://github.com/tenzen-y))
- KEP-2170: Add the TrainJob state transition design ([#2298](https://github.com/kubeflow/training-operator/pull/2298) by [@tenzen-y](https://github.com/tenzen-y))
- KEP-2170: Implement Initializer builders in the JobSet plugin ([#2316](https://github.com/kubeflow/training-operator/pull/2316) by [@andreyvelich](https://github.com/andreyvelich))
- KEP-2170: Implement JobSet, PlainML, and Torch Plugins ([#2308](https://github.com/kubeflow/training-operator/pull/2308) by [@andreyvelich](https://github.com/andreyvelich))
- KEP-2170: Create model and dataset initializers ([#2303](https://github.com/kubeflow/training-operator/pull/2303) by [@andreyvelich](https://github.com/andreyvelich))
- KEP-2170: Generate Python SDK for Kubeflow Training V2 ([#2310](https://github.com/kubeflow/training-operator/pull/2310) by [@andreyvelich](https://github.com/andreyvelich))
- KEP-2170: Initialize runtimes before the manager starts ([#2306](https://github.com/kubeflow/training-operator/pull/2306) by [@tenzen-y](https://github.com/tenzen-y))
- KEP-2170: Strictly verify the CRD marker validation and defaulting in the integration testings ([#2304](https://github.com/kubeflow/training-operator/pull/2304) by [@tenzen-y](https://github.com/tenzen-y))
- KEP-2170: Decouple JobSet from TrainJob ([#2296](https://github.com/kubeflow/training-operator/pull/2296) by [@tenzen-y](https://github.com/tenzen-y))
- KEP-2170: Implement TrainJob Reconciler to manage objects ([#2295](https://github.com/kubeflow/training-operator/pull/2295) by [@tenzen-y](https://github.com/tenzen-y))
- KEP-2170: Add manifests for Kubeflow Training V2 ([#2289](https://github.com/kubeflow/training-operator/pull/2289) by [@andreyvelich](https://github.com/andreyvelich))
- KEP-2170: Adding CEL validations on v2 TrainJob CRD ([#2260](https://github.com/kubeflow/training-operator/pull/2260) by [@akshaychitneni](https://github.com/akshaychitneni))
- KEP-2170: Rename TrainingRuntimeRef to RuntimeRef API ([#2283](https://github.com/kubeflow/training-operator/pull/2283) by [@andreyvelich](https://github.com/andreyvelich))
- KEP-2170: Implement runtime framework ([#2248](https://github.com/kubeflow/training-operator/pull/2248) by [@tenzen-y](https://github.com/tenzen-y))
- [v2alpha] Move GV related codebase ([#2281](https://github.com/kubeflow/training-operator/pull/2281) by [@varshaprasad96](https://github.com/varshaprasad96))
- KEP-2170: Generate clientset, openapi spec for the V2 APIs ([#2273](https://github.com/kubeflow/training-operator/pull/2273) by [@varshaprasad96](https://github.com/varshaprasad96))
- KEP-2170: Implement skeleton webhook servers ([#2251](https://github.com/kubeflow/training-operator/pull/2251) by [@tenzen-y](https://github.com/tenzen-y))
- KEP-2170: Initial Implementations for v2 Manager ([#2236](https://github.com/kubeflow/training-operator/pull/2236) by [@tenzen-y](https://github.com/tenzen-y))
- KEP-2170: Generate CRD manifests for v2 CustomResources ([#2237](https://github.com/kubeflow/training-operator/pull/2237) by [@tenzen-y](https://github.com/tenzen-y))
- KEP-2170: Update Training V2 APIs in the KEP ([#2240](https://github.com/kubeflow/training-operator/pull/2240) by [@andreyvelich](https://github.com/andreyvelich))
- KEP-2170: Add TrainJob and TrainingRuntime APIs ([#2223](https://github.com/kubeflow/training-operator/pull/2223) by [@andreyvelich](https://github.com/andreyvelich))
- KEP-2170: Bind repository into the build environment instead of filecopy ([#2222](https://github.com/kubeflow/training-operator/pull/2222) by [@tenzen-y](https://github.com/tenzen-y))
- KEP-2170: Add directories for the V2 APIs ([#2221](https://github.com/kubeflow/training-operator/pull/2221) by [@andreyvelich](https://github.com/andreyvelich))
- KEP-2170: Add the apiGroup to the TrainingRuntimeRef ([#2201](https://github.com/kubeflow/training-operator/pull/2201) by [@tenzen-y](https://github.com/tenzen-y))
- KEP-2170: Make API specification more restricting ([#2198](https://github.com/kubeflow/training-operator/pull/2198) by [@tenzen-y](https://github.com/tenzen-y))

## Bug Fixes

- [release-1.9] V1: Fix versions in HuggingFace dataset initializer ([#2370](https://github.com/kubeflow/training-operator/pull/2370) by [@google-oss-robot](https://github.com/google-oss-robot))
- Pin accelerate package version in trainer ([#2340](https://github.com/kubeflow/training-operator/pull/2340) by [@gavrissh](https://github.com/gavrissh))
- [fix] Resolve v2alpha API exceptions ([#2317](https://github.com/kubeflow/training-operator/pull/2317) by [@varshaprasad96](https://github.com/varshaprasad96))
- [SDK] Minor fix in wait_for_job_conditions with job_kind python training API ([#2265](https://github.com/kubeflow/training-operator/pull/2265) by [@saileshd1402](https://github.com/saileshd1402))
- [SDK] Fix typo of "get_pvc_spec" ([#2250](https://github.com/kubeflow/training-operator/pull/2250) by [@helenxie-bit](https://github.com/helenxie-bit))
- [Bug] Finish CleanupJob early if the job is suspended. ([#2243](https://github.com/kubeflow/training-operator/pull/2243) by [@mszadkow](https://github.com/mszadkow))
- [SDK] Fix trainer error: Update the version of base image and add "num_labels" for downloading pretrained models ([#2230](https://github.com/kubeflow/training-operator/pull/2230) by [@helenxie-bit](https://github.com/helenxie-bit))
- Update `huggingface_hub` Version in the storage initializer to fix ImportError ([#2180](https://github.com/kubeflow/training-operator/pull/2180) by [@helenxie-bit](https://github.com/helenxie-bit))
- [SDK] Fix Failed condition in wait Job API ([#2160](https://github.com/kubeflow/training-operator/pull/2160) by [@andreyvelich](https://github.com/andreyvelich))
- fix volcano podgroup update issue ([#2079](https://github.com/kubeflow/training-operator/pull/2079) by [@ckyuto](https://github.com/ckyuto))
- [SDK] Fix Incorrect Events in get_job_logs API ([#2122](https://github.com/kubeflow/training-operator/pull/2122) by [@andreyvelich](https://github.com/andreyvelich))

## Misc

- [release-1.9] Add release branch to the image push trigger ([#2377](https://github.com/kubeflow/training-operator/pull/2377) by [@google-oss-robot](https://github.com/google-oss-robot))
- Add e2e test for train API ([#2199](https://github.com/kubeflow/training-operator/pull/2199) by [@helenxie-bit](https://github.com/helenxie-bit))
- buildx link was broken ([#2356](https://github.com/kubeflow/training-operator/pull/2356) by [@Veer0x1](https://github.com/Veer0x1))
- Upgrade helm/kind-action to v1.11.0 ([#2357](https://github.com/kubeflow/training-operator/pull/2357) by [@astefanutti](https://github.com/astefanutti))
- Upgrade Go version to v1.23 ([#2302](https://github.com/kubeflow/training-operator/pull/2302) by [@tenzen-y](https://github.com/tenzen-y))
- Ensure code generation dependencies are downloaded ([#2339](https://github.com/kubeflow/training-operator/pull/2339) by [@astefanutti](https://github.com/astefanutti))
- Added test for create-pytorchjob.ipynb python notebook ([#2274](https://github.com/kubeflow/training-operator/pull/2274) by [@saileshd1402](https://github.com/saileshd1402))
- Remove zw0610 from approvers ([#2343](https://github.com/kubeflow/training-operator/pull/2343) by [@zw0610](https://github.com/zw0610))
- Upgrade kustomization files to Kustomize v5 ([#2326](https://github.com/kubeflow/training-operator/pull/2326) by [@oksanabaza](https://github.com/oksanabaza))
- Add openapi-generator CLI option to skip SDK v2 test generation ([#2338](https://github.com/kubeflow/training-operator/pull/2338) by [@astefanutti](https://github.com/astefanutti))
- Refine the server-side apply installation args ([#2337](https://github.com/kubeflow/training-operator/pull/2337) by [@tenzen-y](https://github.com/tenzen-y))
- Ignore cache exporting errors in the image building workflows ([#2336](https://github.com/kubeflow/training-operator/pull/2336) by [@tenzen-y](https://github.com/tenzen-y))
- Pin Gloo repository in JAX Dockerfile to a specific commit ([#2329](https://github.com/kubeflow/training-operator/pull/2329) by [@sandipanpanda](https://github.com/sandipanpanda))
- Update tf job examples to tf v2 ([#2270](https://github.com/kubeflow/training-operator/pull/2270) by [@YosiElias](https://github.com/YosiElias))
- Remove Prometheus Monitoring doc ([#2301](https://github.com/kubeflow/training-operator/pull/2301) by [@sophie0730](https://github.com/sophie0730))
- Upgrade Deepspeed demo dependencies ([#2294](https://github.com/kubeflow/training-operator/pull/2294) by [@Syulin7](https://github.com/Syulin7))
- [SDK] test: add unit test for list_jobs method of the training_client ([#2267](https://github.com/kubeflow/training-operator/pull/2267) by [@seanlaii](https://github.com/seanlaii))
- [SDK] Training Client Conditions related unit tests ([#2253](https://github.com/kubeflow/training-operator/pull/2253) by [@Bobbins228](https://github.com/Bobbins228))
- [SDK] test: add unit test for get_job_logs method of the training_client ([#2275](https://github.com/kubeflow/training-operator/pull/2275) by [@seanlaii](https://github.com/seanlaii))
- [SDK] test: add unit test for get_job method of the training_client ([#2205](https://github.com/kubeflow/training-operator/pull/2205) by [@Bobbins228](https://github.com/Bobbins228))
- [SDK] test: add unit tests for delete_job() method ([#2232](https://github.com/kubeflow/training-operator/pull/2232) by [@Bobbins228](https://github.com/Bobbins228))
- [SDK] Add UTs for `wait_for_job_conditions` ([#2196](https://github.com/kubeflow/training-operator/pull/2196) by [@Electronic-Waste](https://github.com/Electronic-Waste))
- [SDK] Unit tests for TrainingClient APIs - get_job_pod_names and update_job ([#2192](https://github.com/kubeflow/training-operator/pull/2192) by [@YosiElias](https://github.com/YosiElias))
- [SDK] Add more unit tests for TrainingClient APIs - get_job_pods ([#2175](https://github.com/kubeflow/training-operator/pull/2175) by [@YosiElias](https://github.com/YosiElias))
- Update JAX image to use image published by Kubeflow ([#2264](https://github.com/kubeflow/training-operator/pull/2264) by [@sandipanpanda](https://github.com/sandipanpanda))
- Update README and out-of-date docs ([#2252](https://github.com/kubeflow/training-operator/pull/2252) by [@andreyvelich](https://github.com/andreyvelich))
- Clean up Go modules ([#2238](https://github.com/kubeflow/training-operator/pull/2238) by [@tenzen-y](https://github.com/tenzen-y))
- Change isort profile to black for full compatibility ([#2234](https://github.com/kubeflow/training-operator/pull/2234) by [@Ygnas](https://github.com/Ygnas))
- Enhance pre-commit hooks with flake8 linting ([#2195](https://github.com/kubeflow/training-operator/pull/2195) by [@Ygnas](https://github.com/Ygnas))
- Implement pre-commit hooks ([#2184](https://github.com/kubeflow/training-operator/pull/2184) by [@droctothorpe](https://github.com/droctothorpe))
- Add command to re-run GitHub Actions tests ([#2167](https://github.com/kubeflow/training-operator/pull/2167) by [@andreyvelich](https://github.com/andreyvelich))
- Update JAX integration proposal ([#2165](https://github.com/kubeflow/training-operator/pull/2165) by [@sandipanpanda](https://github.com/sandipanpanda))
- Update release document ([#2153](https://github.com/kubeflow/training-operator/pull/2153) by [@andreyvelich](https://github.com/andreyvelich))
- update volcano to v1.9.0 ([#2148](https://github.com/kubeflow/training-operator/pull/2148) by [@lowang-bh](https://github.com/lowang-bh))
- Update Slack Invitation ([#2142](https://github.com/kubeflow/training-operator/pull/2142) by [@andreyvelich](https://github.com/andreyvelich))
- Refine the integration tests for the immutable PyTorchJob queueName ([#2130](https://github.com/kubeflow/training-operator/pull/2130) by [@tenzen-y](https://github.com/tenzen-y))
- Add GitHub Issue Template ([#2129](https://github.com/kubeflow/training-operator/pull/2129) by [@andreyvelich](https://github.com/andreyvelich))
- Update the images to the latest tag in master branch ([#2128](https://github.com/kubeflow/training-operator/pull/2128) by [@johnugeorge](https://github.com/johnugeorge))
- Updated Github Action Workflows as per issue #2117 ([#2123](https://github.com/kubeflow/training-operator/pull/2123) by [@hkiiita](https://github.com/hkiiita))
- changed package name to flake8 to fix pytests pip install ([#2109](https://github.com/kubeflow/training-operator/pull/2109) by [@ChristopheBrown](https://github.com/ChristopheBrown))
- chore(fix): isort xgboost ([#2098](https://github.com/kubeflow/training-operator/pull/2098) by [@harshithbelagur](https://github.com/harshithbelagur))
- Fix isort on examples/pytorch ([#2094](https://github.com/kubeflow/training-operator/pull/2094) by [@marcmaliar](https://github.com/marcmaliar))

[Full Changelog](https://github.com/kubeflow/training-operator/compare/v1.8.1...v1.9.0)

# [v1.9.0-rc.0](https://github.com/kubeflow/training-operator/tree/v1.9.0-rc.0) (2025-01-07)

## Breaking Changes

- Upgrade Kubernetes to v1.31.3 ([#2330](https://github.com/kubeflow/training-operator/pull/2330) by [@astefanutti](https://github.com/astefanutti))
- Upgrade Kubernetes to v1.30.7 ([#2332](https://github.com/kubeflow/training-operator/pull/2332) by [@astefanutti](https://github.com/astefanutti))
- Update the name of PVC in `train` API ([#2187](https://github.com/kubeflow/training-operator/pull/2187) by [@helenxie-bit](https://github.com/helenxie-bit))
- Remove support for MXJob ([#2150](https://github.com/kubeflow/training-operator/pull/2150) by [@tariq-hasan](https://github.com/tariq-hasan))
- Support Python 3.11 and Drop Python 3.7 ([#2105](https://github.com/kubeflow/training-operator/pull/2105) by [@tenzen-y](https://github.com/tenzen-y))

## New Features

### Distributed JAX

- Add JAX controller ([#2194](https://github.com/kubeflow/training-operator/pull/2194) by [@sandipanpanda](https://github.com/sandipanpanda))
- Add JAX API ([#2163](https://github.com/kubeflow/training-operator/pull/2163) by [@sandipanpanda](https://github.com/sandipanpanda))
- JAX Integration Enhancement Proposal ([#2125](https://github.com/kubeflow/training-operator/pull/2125) by [@sandipanpanda](https://github.com/sandipanpanda))

### New Examples

- FSDP Example for T5 Fine-Tuning and PyTorchJob ([#2286](https://github.com/kubeflow/training-operator/pull/2286) by [@andreyvelich](https://github.com/andreyvelich))
- Add DeepSpeed Example with Pytorch Operator ([#2235](https://github.com/kubeflow/training-operator/pull/2235) by [@Syulin7](https://github.com/Syulin7))

### Control Plane Updates

- Validate pytorchjob workers are configured when elasticpolicy is configured ([#2320](https://github.com/kubeflow/training-operator/pull/2320) by [@tarat44](https://github.com/tarat44))
- [Feature] Support managed by external controller ([#2203](https://github.com/kubeflow/training-operator/pull/2203) by [@mszadkow](https://github.com/mszadkow))
- Update trainer to ensure type consistency for `train_args` and `lora_config` ([#2181](https://github.com/kubeflow/training-operator/pull/2181) by [@helenxie-bit](https://github.com/helenxie-bit))
- Support ARM64 platform in TensorFlow examples ([#2119](https://github.com/kubeflow/training-operator/pull/2119) by [@akhilsaivenkata](https://github.com/akhilsaivenkata))
- Feat: Support ARM64 platform in XGBoost examples ([#2114](https://github.com/kubeflow/training-operator/pull/2114) by [@tico88612](https://github.com/tico88612))
- ARM64 supported in PyTorch examples ([#2116](https://github.com/kubeflow/training-operator/pull/2116) by [@danielsuh05](https://github.com/danielsuh05))

### SDK Updates

- [SDK] Adding env vars ([#2285](https://github.com/kubeflow/training-operator/pull/2285) by [@tarekabouzeid](https://github.com/tarekabouzeid))
- [SDK] Use torchrun to create PyTorchJob from function ([#2276](https://github.com/kubeflow/training-operator/pull/2276) by [@andreyvelich](https://github.com/andreyvelich))
- [SDK] move env var to constants.py ([#2268](https://github.com/kubeflow/training-operator/pull/2268) by [@varshaprasad96](https://github.com/varshaprasad96))
- [SDK] Allow customising base trainer and storage images in Train API ([#2261](https://github.com/kubeflow/training-operator/pull/2261) by [@varshaprasad96](https://github.com/varshaprasad96))
- [SDK] Read namespace from the current context ([#2255](https://github.com/kubeflow/training-operator/pull/2255) by [@andreyvelich](https://github.com/andreyvelich))
- [SDK] Sync Transformers version for train API ([#2146](https://github.com/kubeflow/training-operator/pull/2146) by [@andreyvelich](https://github.com/andreyvelich))
- [SDK] Explain Python version support cycle ([#2144](https://github.com/kubeflow/training-operator/pull/2144) by [@andreyvelich](https://github.com/andreyvelich))

### Kubeflow Training V2

- KEP-2170: Kubeflow Training V2 API ([#2171](https://github.com/kubeflow/training-operator/pull/2171) by [@andreyvelich](https://github.com/andreyvelich))
- KEP-2170: Update V2 KEP with MPI Runtime info ([#2345](https://github.com/kubeflow/training-operator/pull/2345) by [@andreyvelich](https://github.com/andreyvelich))
- Always update TrainJob status on errors ([#2352](https://github.com/kubeflow/training-operator/pull/2352) by [@astefanutti](https://github.com/astefanutti))
- Fix TrainJob status comparison and update ([#2353](https://github.com/kubeflow/training-operator/pull/2353) by [@astefanutti](https://github.com/astefanutti))
- Add required RBAC on TrainJob finalizer sub-resources ([#2350](https://github.com/kubeflow/training-operator/pull/2350) by [@astefanutti](https://github.com/astefanutti))
- KEP-2170: [SDK] Initial implementation of the Kubeflow Training V2 Python SDK ([#2324](https://github.com/kubeflow/training-operator/pull/2324) by [@andreyvelich](https://github.com/andreyvelich))
- KEP-2170: Add Torch Distributed Runtime ([#2328](https://github.com/kubeflow/training-operator/pull/2328) by [@andreyvelich](https://github.com/andreyvelich))
- KEP-2170: Add TrainJob conditions ([#2322](https://github.com/kubeflow/training-operator/pull/2322) by [@tenzen-y](https://github.com/tenzen-y))
- KEP-2170: Add the TrainJob state transition design ([#2298](https://github.com/kubeflow/training-operator/pull/2298) by [@tenzen-y](https://github.com/tenzen-y))
- KEP-2170: Implement Initializer builders in the JobSet plugin ([#2316](https://github.com/kubeflow/training-operator/pull/2316) by [@andreyvelich](https://github.com/andreyvelich))
- KEP-2170: Implement JobSet, PlainML, and Torch Plugins ([#2308](https://github.com/kubeflow/training-operator/pull/2308) by [@andreyvelich](https://github.com/andreyvelich))
- KEP-2170: Create model and dataset initializers ([#2303](https://github.com/kubeflow/training-operator/pull/2303) by [@andreyvelich](https://github.com/andreyvelich))
- KEP-2170: Generate Python SDK for Kubeflow Training V2 ([#2310](https://github.com/kubeflow/training-operator/pull/2310) by [@andreyvelich](https://github.com/andreyvelich))
- KEP-2170: Initialize runtimes before the manager starts ([#2306](https://github.com/kubeflow/training-operator/pull/2306) by [@tenzen-y](https://github.com/tenzen-y))
- KEP-2170: Strictly verify the CRD marker validation and defaulting in the integration testings ([#2304](https://github.com/kubeflow/training-operator/pull/2304) by [@tenzen-y](https://github.com/tenzen-y))
- KEP-2170: Decouple JobSet from TrainJob ([#2296](https://github.com/kubeflow/training-operator/pull/2296) by [@tenzen-y](https://github.com/tenzen-y))
- KEP-2170: Implement TrainJob Reconciler to manage objects ([#2295](https://github.com/kubeflow/training-operator/pull/2295) by [@tenzen-y](https://github.com/tenzen-y))
- KEP-2170: Add manifests for Kubeflow Training V2 ([#2289](https://github.com/kubeflow/training-operator/pull/2289) by [@andreyvelich](https://github.com/andreyvelich))
- KEP-2170: Adding CEL validations on v2 TrainJob CRD ([#2260](https://github.com/kubeflow/training-operator/pull/2260) by [@akshaychitneni](https://github.com/akshaychitneni))
- KEP-2170: Rename TrainingRuntimeRef to RuntimeRef API ([#2283](https://github.com/kubeflow/training-operator/pull/2283) by [@andreyvelich](https://github.com/andreyvelich))
- KEP-2170: Implement runtime framework ([#2248](https://github.com/kubeflow/training-operator/pull/2248) by [@tenzen-y](https://github.com/tenzen-y))
- [v2alpha] Move GV related codebase ([#2281](https://github.com/kubeflow/training-operator/pull/2281) by [@varshaprasad96](https://github.com/varshaprasad96))
- KEP-2170: Generate clientset, openapi spec for the V2 APIs ([#2273](https://github.com/kubeflow/training-operator/pull/2273) by [@varshaprasad96](https://github.com/varshaprasad96))
- KEP-2170: Implement skeleton webhook servers ([#2251](https://github.com/kubeflow/training-operator/pull/2251) by [@tenzen-y](https://github.com/tenzen-y))
- KEP-2170: Initial Implementations for v2 Manager ([#2236](https://github.com/kubeflow/training-operator/pull/2236) by [@tenzen-y](https://github.com/tenzen-y))
- KEP-2170: Generate CRD manifests for v2 CustomResources ([#2237](https://github.com/kubeflow/training-operator/pull/2237) by [@tenzen-y](https://github.com/tenzen-y))
- KEP-2170: Update Training V2 APIs in the KEP ([#2240](https://github.com/kubeflow/training-operator/pull/2240) by [@andreyvelich](https://github.com/andreyvelich))
- KEP-2170: Add TrainJob and TrainingRuntime APIs ([#2223](https://github.com/kubeflow/training-operator/pull/2223) by [@andreyvelich](https://github.com/andreyvelich))
- KEP-2170: Bind repository into the build environment instead of filecopy ([#2222](https://github.com/kubeflow/training-operator/pull/2222) by [@tenzen-y](https://github.com/tenzen-y))
- KEP-2170: Add directories for the V2 APIs ([#2221](https://github.com/kubeflow/training-operator/pull/2221) by [@andreyvelich](https://github.com/andreyvelich))
- KEP-2170: Add the apiGroup to the TrainingRuntimeRef ([#2201](https://github.com/kubeflow/training-operator/pull/2201) by [@tenzen-y](https://github.com/tenzen-y))
- KEP-2170: Make API specification more restricting ([#2198](https://github.com/kubeflow/training-operator/pull/2198) by [@tenzen-y](https://github.com/tenzen-y))

## Bug Fixes

- [release-1.9] V1: Fix versions in HuggingFace dataset initializer ([#2370](https://github.com/kubeflow/training-operator/pull/2370) by [@google-oss-robot](https://github.com/google-oss-robot))
- Pin accelerate package version in trainer ([#2340](https://github.com/kubeflow/training-operator/pull/2340) by [@gavrissh](https://github.com/gavrissh))
- [fix] Resolve v2alpha API exceptions ([#2317](https://github.com/kubeflow/training-operator/pull/2317) by [@varshaprasad96](https://github.com/varshaprasad96))
- [SDK] Minor fix in wait_for_job_conditions with job_kind python training API ([#2265](https://github.com/kubeflow/training-operator/pull/2265) by [@saileshd1402](https://github.com/saileshd1402))
- [SDK] Fix typo of "get_pvc_spec" ([#2250](https://github.com/kubeflow/training-operator/pull/2250) by [@helenxie-bit](https://github.com/helenxie-bit))
- [Bug] Finish CleanupJob early if the job is suspended. ([#2243](https://github.com/kubeflow/training-operator/pull/2243) by [@mszadkow](https://github.com/mszadkow))
- [SDK] Fix trainer error: Update the version of base image and add "num_labels" for downloading pretrained models ([#2230](https://github.com/kubeflow/training-operator/pull/2230) by [@helenxie-bit](https://github.com/helenxie-bit))
- Update `huggingface_hub` Version in the storage initializer to fix ImportError ([#2180](https://github.com/kubeflow/training-operator/pull/2180) by [@helenxie-bit](https://github.com/helenxie-bit))
- [SDK] Fix Failed condition in wait Job API ([#2160](https://github.com/kubeflow/training-operator/pull/2160) by [@andreyvelich](https://github.com/andreyvelich))
- fix volcano podgroup update issue ([#2079](https://github.com/kubeflow/training-operator/pull/2079) by [@ckyuto](https://github.com/ckyuto))
- [SDK] Fix Incorrect Events in get_job_logs API ([#2122](https://github.com/kubeflow/training-operator/pull/2122) by [@andreyvelich](https://github.com/andreyvelich))

## Misc

- [release-1.9] Add release branch to the image push trigger ([#2377](https://github.com/kubeflow/training-operator/pull/2377) by [@google-oss-robot](https://github.com/google-oss-robot))
- Add e2e test for train API ([#2199](https://github.com/kubeflow/training-operator/pull/2199) by [@helenxie-bit](https://github.com/helenxie-bit))
- buildx link was broken ([#2356](https://github.com/kubeflow/training-operator/pull/2356) by [@Veer0x1](https://github.com/Veer0x1))
- Upgrade helm/kind-action to v1.11.0 ([#2357](https://github.com/kubeflow/training-operator/pull/2357) by [@astefanutti](https://github.com/astefanutti))
- Upgrade Go version to v1.23 ([#2302](https://github.com/kubeflow/training-operator/pull/2302) by [@tenzen-y](https://github.com/tenzen-y))
- Ensure code generation dependencies are downloaded ([#2339](https://github.com/kubeflow/training-operator/pull/2339) by [@astefanutti](https://github.com/astefanutti))
- Added test for create-pytorchjob.ipynb python notebook ([#2274](https://github.com/kubeflow/training-operator/pull/2274) by [@saileshd1402](https://github.com/saileshd1402))
- Remove zw0610 from approvers ([#2343](https://github.com/kubeflow/training-operator/pull/2343) by [@zw0610](https://github.com/zw0610))
- Upgrade kustomization files to Kustomize v5 ([#2326](https://github.com/kubeflow/training-operator/pull/2326) by [@oksanabaza](https://github.com/oksanabaza))
- Add openapi-generator CLI option to skip SDK v2 test generation ([#2338](https://github.com/kubeflow/training-operator/pull/2338) by [@astefanutti](https://github.com/astefanutti))
- Refine the server-side apply installation args ([#2337](https://github.com/kubeflow/training-operator/pull/2337) by [@tenzen-y](https://github.com/tenzen-y))
- Ignore cache exporting errors in the image building workflows ([#2336](https://github.com/kubeflow/training-operator/pull/2336) by [@tenzen-y](https://github.com/tenzen-y))
- Pin Gloo repository in JAX Dockerfile to a specific commit ([#2329](https://github.com/kubeflow/training-operator/pull/2329) by [@sandipanpanda](https://github.com/sandipanpanda))
- Update tf job examples to tf v2 ([#2270](https://github.com/kubeflow/training-operator/pull/2270) by [@YosiElias](https://github.com/YosiElias))
- Remove Prometheus Monitoring doc ([#2301](https://github.com/kubeflow/training-operator/pull/2301) by [@sophie0730](https://github.com/sophie0730))
- Upgrade Deepspeed demo dependencies ([#2294](https://github.com/kubeflow/training-operator/pull/2294) by [@Syulin7](https://github.com/Syulin7))
- [SDK] test: add unit test for list_jobs method of the training_client ([#2267](https://github.com/kubeflow/training-operator/pull/2267) by [@seanlaii](https://github.com/seanlaii))
- [SDK] Training Client Conditions related unit tests ([#2253](https://github.com/kubeflow/training-operator/pull/2253) by [@Bobbins228](https://github.com/Bobbins228))
- [SDK] test: add unit test for get_job_logs method of the training_client ([#2275](https://github.com/kubeflow/training-operator/pull/2275) by [@seanlaii](https://github.com/seanlaii))
- [SDK] test: add unit test for get_job method of the training_client ([#2205](https://github.com/kubeflow/training-operator/pull/2205) by [@Bobbins228](https://github.com/Bobbins228))
- [SDK] test: add unit tests for delete_job() method ([#2232](https://github.com/kubeflow/training-operator/pull/2232) by [@Bobbins228](https://github.com/Bobbins228))
- [SDK] Add UTs for `wait_for_job_conditions` ([#2196](https://github.com/kubeflow/training-operator/pull/2196) by [@Electronic-Waste](https://github.com/Electronic-Waste))
- [SDK] Unit tests for TrainingClient APIs - get_job_pod_names and update_job ([#2192](https://github.com/kubeflow/training-operator/pull/2192) by [@YosiElias](https://github.com/YosiElias))
- [SDK] Add more unit tests for TrainingClient APIs - get_job_pods ([#2175](https://github.com/kubeflow/training-operator/pull/2175) by [@YosiElias](https://github.com/YosiElias))
- Update JAX image to use image published by Kubeflow ([#2264](https://github.com/kubeflow/training-operator/pull/2264) by [@sandipanpanda](https://github.com/sandipanpanda))
- Update README and out-of-date docs ([#2252](https://github.com/kubeflow/training-operator/pull/2252) by [@andreyvelich](https://github.com/andreyvelich))
- Clean up Go modules ([#2238](https://github.com/kubeflow/training-operator/pull/2238) by [@tenzen-y](https://github.com/tenzen-y))
- Change isort profile to black for full compatibility ([#2234](https://github.com/kubeflow/training-operator/pull/2234) by [@Ygnas](https://github.com/Ygnas))
- Enhance pre-commit hooks with flake8 linting ([#2195](https://github.com/kubeflow/training-operator/pull/2195) by [@Ygnas](https://github.com/Ygnas))
- Implement pre-commit hooks ([#2184](https://github.com/kubeflow/training-operator/pull/2184) by [@droctothorpe](https://github.com/droctothorpe))
- Add command to re-run GitHub Actions tests ([#2167](https://github.com/kubeflow/training-operator/pull/2167) by [@andreyvelich](https://github.com/andreyvelich))
- Update JAX integration proposal ([#2165](https://github.com/kubeflow/training-operator/pull/2165) by [@sandipanpanda](https://github.com/sandipanpanda))
- Update release document ([#2153](https://github.com/kubeflow/training-operator/pull/2153) by [@andreyvelich](https://github.com/andreyvelich))
- update volcano to v1.9.0 ([#2148](https://github.com/kubeflow/training-operator/pull/2148) by [@lowang-bh](https://github.com/lowang-bh))
- Update Slack Invitation ([#2142](https://github.com/kubeflow/training-operator/pull/2142) by [@andreyvelich](https://github.com/andreyvelich))
- Refine the integration tests for the immutable PyTorchJob queueName ([#2130](https://github.com/kubeflow/training-operator/pull/2130) by [@tenzen-y](https://github.com/tenzen-y))
- Add GitHub Issue Template ([#2129](https://github.com/kubeflow/training-operator/pull/2129) by [@andreyvelich](https://github.com/andreyvelich))
- Update the images to the latest tag in master branch ([#2128](https://github.com/kubeflow/training-operator/pull/2128) by [@johnugeorge](https://github.com/johnugeorge))
- Updated Github Action Workflows as per issue #2117 ([#2123](https://github.com/kubeflow/training-operator/pull/2123) by [@hkiiita](https://github.com/hkiiita))
- changed package name to flake8 to fix pytests pip install ([#2109](https://github.com/kubeflow/training-operator/pull/2109) by [@ChristopheBrown](https://github.com/ChristopheBrown))
- chore(fix): isort xgboost ([#2098](https://github.com/kubeflow/training-operator/pull/2098) by [@harshithbelagur](https://github.com/harshithbelagur))
- Fix isort on examples/pytorch ([#2094](https://github.com/kubeflow/training-operator/pull/2094) by [@marcmaliar](https://github.com/marcmaliar))

[Full Changelog](https://github.com/kubeflow/training-operator/compare/v1.8.1...v1.9.0-rc.0)

# [v1.8.1](https://github.com/kubeflow/training-operator/tree/v1.8.1) (2024-09-10)

## Bug Fixes

- [Bug] Finish CleanupJob early if the job is suspended ([#2243](https://github.com/kubeflow/training-operator/pull/2243) by [@mszadkow](https://github.com/mszadkow))
- [SDK] Fix trainer error: Update the version of base image and add "num_labels" for downloading pretrained models ([#2230](https://github.com/kubeflow/training-operator/pull/2230) by [@helenxie-bit](https://github.com/helenxie-bit))
- Update huggingface_hub Version in the storage initializer to fix ImportError ([#2180](https://github.com/kubeflow/training-operator/pull/2180) by [@helenxie-bit](https://github.com/helenxie-bit))

[Full Changelog](https://github.com/kubeflow/training-operator/compare/v1.8.0...v1.8.1)

# [v1.8.0](https://github.com/kubeflow/training-operator/tree/v1.8.0) (2024-07-15)

## Breaking Changes

- [SDK] Support Python 3.11 and Drop Python 3.7 ([#2105](https://github.com/kubeflow/training-operator/pull/2105) by [@tenzen-y](https://github.com/tenzen-y))
- Support K8s v1.29 and Drop K8s v1.26 ([#2039](https://github.com/kubeflow/training-operator/pull/2039) by [@tenzen-y](https://github.com/tenzen-y))
- Support K8s v1.28 and Drop K8s v1.25 ([#2038](https://github.com/kubeflow/training-operator/pull/2038) by [@tenzen-y](https://github.com/tenzen-y))
- Deprecation Notice for MXJob ([#2058](https://github.com/kubeflow/training-operator/pull/2058) by [@tenzen-y](https://github.com/tenzen-y))
- ⚠️ Breaking Changes: Rename `monitoring-port` flag to `webook-server-port` ([#1925](https://github.com/kubeflow/training-operator/pull/1925) by [@afritzler](https://github.com/afritzler))

## New Features

### LLM Fine-Tuning API

- Train/Fine-tune API Proposal for LLMs ([#1945](https://github.com/kubeflow/training-operator/pull/1945) by [@deepanker13](https://github.com/deepanker13))
- [SDK] Train API for LLM Fine-Tuning ([#1962](https://github.com/kubeflow/training-operator/pull/1962) by [@deepanker13](https://github.com/deepanker13))
- Modify LLM Trainer to support BERT and Tiny LLaMA ([#2031](https://github.com/kubeflow/training-operator/pull/2031) by [@andreyvelich](https://github.com/andreyvelich))
- Support arm64 for Hugging Face trainer ([#2028](https://github.com/kubeflow/training-operator/pull/2028) by [@tariq-hasan](https://github.com/tariq-hasan))
- Add Fine-Tune BERT LLM Example ([#2021](https://github.com/kubeflow/training-operator/pull/2021) by [@andreyvelich](https://github.com/andreyvelich))
- Train api dataset download changes ([#1959](https://github.com/kubeflow/training-operator/pull/1959) by [@deepanker13](https://github.com/deepanker13))
- Train api init container creation ([#1958](https://github.com/kubeflow/training-operator/pull/1958) by [@deepanker13](https://github.com/deepanker13))
- [SDK] Add docstring for Train API ([#2075](https://github.com/kubeflow/training-operator/pull/2075) by [@andreyvelich](https://github.com/andreyvelich))

### Control Plane Updates

- Upgrade scheduler-plugins to v0.28.9 ([#2065](https://github.com/kubeflow/training-operator/pull/2065) by [@tenzen-y](https://github.com/tenzen-y))
- Implement webhook validations for the PaddleJob ([#2057](https://github.com/kubeflow/training-operator/pull/2057) by [@tenzen-y](https://github.com/tenzen-y))
- Implement webhook validations for the XGBoostJob ([#2052](https://github.com/kubeflow/training-operator/pull/2052) by [@tenzen-y](https://github.com/tenzen-y))
- Implement webhook validation for the TFJob ([#2051](https://github.com/kubeflow/training-operator/pull/2051) by [@tenzen-y](https://github.com/tenzen-y))
- Implement webhook validations for the PyTorchJob ([#2035](https://github.com/kubeflow/training-operator/pull/2035) by [@tenzen-y](https://github.com/tenzen-y))
- Upgrade PyTorchJob examples to PyTorch v2 ([#2024](https://github.com/kubeflow/training-operator/pull/2024) by [@champon1020](https://github.com/champon1020))
- Upgrade Go version to v1.22 ([#2046](https://github.com/kubeflow/training-operator/pull/2046) by [@tenzen-y](https://github.com/tenzen-y))

### SDK Improvements

- [SDK] Add resources per worker for Create Job API ([#1990](https://github.com/kubeflow/training-operator/pull/1990) by [@andreyvelich](https://github.com/andreyvelich))
- [SDK] Fix Worker and Master templates for PyTorchJob ([#1988](https://github.com/kubeflow/training-operator/pull/1988) by [@andreyvelich](https://github.com/andreyvelich))
- [SDK] Get Kubernetes Events for Job ([#1975](https://github.com/kubeflow/training-operator/pull/1975) by [@andreyvelich](https://github.com/andreyvelich))
- SDK: Upgrade the minimum required Kubernetes version to v1.27.2 ([#2066](https://github.com/kubeflow/training-operator/pull/2066) by [@tenzen-y](https://github.com/tenzen-y))
- [SDK] Add information about TrainingClient logging ([#1973](https://github.com/kubeflow/training-operator/pull/1973) by [@andreyvelich](https://github.com/andreyvelich))
- Training operator SDK unit test ([#1938](https://github.com/kubeflow/training-operator/pull/1938) by [@deepanker13](https://github.com/deepanker13))
- [SDK] Consolidate Naming for CRUD APIs ([#1907](https://github.com/kubeflow/training-operator/pull/1907) by [@andreyvelich](https://github.com/andreyvelich))

## Bug Fixes

- [SDK] Fix Failed condition in wait Job API ([#2160](https://github.com/kubeflow/training-operator/pull/2160) by [@andreyvelich](https://github.com/andreyvelich))
- [SDK] Sync Transformers version for train API ([#2147](https://github.com/kubeflow/training-operator/pull/2147) by [@andreyvelich](https://github.com/andreyvelich))
- [SDK] Changed package name to flake8 to fix pip install ([#2140](https://github.com/kubeflow/training-operator/pull/2140) by [@tenzen-y](https://github.com/tenzen-y))
- [SDK] Fix Incorrect Events in get_job_logs API ([#2138](https://github.com/kubeflow/training-operator/pull/2138) by [@tenzen-y](https://github.com/tenzen-y))
- Fix volcano podgroup update issue ([#2079](https://github.com/kubeflow/training-operator/pull/2079) by [@ckyuto](https://github.com/ckyuto))
- Fix import for HuggingFace Dataset Provider ([#2085](https://github.com/kubeflow/training-operator/pull/2085) by [@andreyvelich](https://github.com/andreyvelich))
- Updated examples for train API ([#2077](https://github.com/kubeflow/training-operator/pull/2077) by [@shruti2522](https://github.com/shruti2522))
- Fail job for non-retryable exit codes ([#2071](https://github.com/kubeflow/training-operator/pull/2071) by [@kellyaa](https://github.com/kellyaa))
- E2E: Replace outdated images with latest ones ([#2083](https://github.com/kubeflow/training-operator/pull/2083) by [@tenzen-y](https://github.com/tenzen-y))
- fix wrong filepath in the simple example command ([#2062](https://github.com/kubeflow/training-operator/pull/2062) by [@qzoscar](https://github.com/qzoscar))
- fix(example): add installation of python-etcd in Pytorch example ([#2064](https://github.com/kubeflow/training-operator/pull/2064) by [@champon1020](https://github.com/champon1020))
- fix: Upgrade controller-gen to v0.14.0 ([#2026](https://github.com/kubeflow/training-operator/pull/2026) by [@champon1020](https://github.com/champon1020))
- Fix build workflow config for pytorch-torchrun-example ([#2020](https://github.com/kubeflow/training-operator/pull/2020) by [@PeterWrighten](https://github.com/PeterWrighten))
- Fix Distributed Data Samplers in PyTorch Examples ([#2012](https://github.com/kubeflow/training-operator/pull/2012) by [@andreyvelich](https://github.com/andreyvelich))
- Fix URL in python SDK setup.py ([#2011](https://github.com/kubeflow/training-operator/pull/2011) by [@garymm](https://github.com/garymm))
- Fix for Github CI to publish HF trainer image ([#1987](https://github.com/kubeflow/training-operator/pull/1987) by [@johnugeorge](https://github.com/johnugeorge))
- train api jupyternotebook fix ([#1984](https://github.com/kubeflow/training-operator/pull/1984) by [@deepanker13](https://github.com/deepanker13))
- fix: volcano podgroup should has a non-empty queue name ([#1977](https://github.com/kubeflow/training-operator/pull/1977) by [@lowang-bh](https://github.com/lowang-bh))
- Fix Master Label for PyTorchJob ([#1974](https://github.com/kubeflow/training-operator/pull/1974) by [@andreyvelich](https://github.com/andreyvelich))
- IsMasterRole fix in pytorchjob controller ([#1969](https://github.com/kubeflow/training-operator/pull/1969) by [@deepanker13](https://github.com/deepanker13))
- [fix] replace ${go env GOPATH} with $(go env GOPATH) to get the prope… ([#1952](https://github.com/kubeflow/training-operator/pull/1952) by [@double12gzh](https://github.com/double12gzh))
- Fixing issues with providing existing service account ([#1918](https://github.com/kubeflow/training-operator/pull/1918) by [@rpemsel](https://github.com/rpemsel))

## Misc

- Refine the integration tests for the immutable PyTorchJob ([#2130](https://github.com/kubeflow/training-operator/pull/2130) by [@tenzen-y](https://github.com/tenzen-y))
- Update training operator image to latest ([#2089](https://github.com/kubeflow/training-operator/pull/2089) by [@johnugeorge](https://github.com/johnugeorge))
- Update sdk to v1.8.0rc0 ([#2087](https://github.com/kubeflow/training-operator/pull/2087) by [@johnugeorge](https://github.com/johnugeorge))
- Test: Simplify and Identify pod-controller envtest ([#2084](https://github.com/kubeflow/training-operator/pull/2084) by [@tenzen-y](https://github.com/tenzen-y))
- Remove deadcode related to PodDisruptionBudget ([#2073](https://github.com/kubeflow/training-operator/pull/2073) by [@tenzen-y](https://github.com/tenzen-y))
- docs: updating docs for local development ([#2074](https://github.com/kubeflow/training-operator/pull/2074) by [@franciscojavierarceo](https://github.com/franciscojavierarceo))
- PyTorchJob: Always show warnings when using elasticPolicy.nProcPerNode ([#2067](https://github.com/kubeflow/training-operator/pull/2067) by [@tenzen-y](https://github.com/tenzen-y))
- Updated developer docs to include Kind ([#2061](https://github.com/kubeflow/training-operator/pull/2061) by [@franciscojavierarceo](https://github.com/franciscojavierarceo))
- adding fine tune example with s3 as the dataset store ([#2006](https://github.com/kubeflow/training-operator/pull/2006) by [@deepanker13](https://github.com/deepanker13))
- CI: Use a mode=min in the builder cache ([#2053](https://github.com/kubeflow/training-operator/pull/2053) by [@tenzen-y](https://github.com/tenzen-y))
- Fix: upgrade version of crd-ref-docs, which caused panic with go v1.22 ([#2043](https://github.com/kubeflow/training-operator/pull/2043) by [@jdcfd](https://github.com/jdcfd))
- Remove Dockerfile.ppc64le of pytorch example ([#2042](https://github.com/kubeflow/training-operator/pull/2042) by [@champon1020](https://github.com/champon1020))
- publish torchrun example via Dockerfile ([#2018](https://github.com/kubeflow/training-operator/pull/2018) by [@PeterWrighten](https://github.com/PeterWrighten))
- Updated examples/pytorch to disable istio sidecar injection ([#2004](https://github.com/kubeflow/training-operator/pull/2004) by [@jdcfd](https://github.com/jdcfd))
- [docs] development guide update ([#1995](https://github.com/kubeflow/training-operator/pull/1995) by [@shashank-iitbhu](https://github.com/shashank-iitbhu))
- Add Kubeflow Website links to README ([#1983](https://github.com/kubeflow/training-operator/pull/1983) by [@andreyvelich](https://github.com/andreyvelich))
- publish trainer hugging face image ([#1985](https://github.com/kubeflow/training-operator/pull/1985) by [@deepanker13](https://github.com/deepanker13))
- Adding Training image needed for train api ([#1963](https://github.com/kubeflow/training-operator/pull/1963) by [@deepanker13](https://github.com/deepanker13))
- Add test to create PyTorchJob from func ([#1979](https://github.com/kubeflow/training-operator/pull/1979) by [@andreyvelich](https://github.com/andreyvelich))
- Corrected Some Spelling And Grammatical Errors ([#1980](https://github.com/kubeflow/training-operator/pull/1980) by [@daniel-hutao](https://github.com/daniel-hutao))
- torchrun example with cpu version pytorch ([#1965](https://github.com/kubeflow/training-operator/pull/1965) by [@kuizhiqing](https://github.com/kuizhiqing))
- utils changes needed to add train api ([#1954](https://github.com/kubeflow/training-operator/pull/1954) by [@deepanker13](https://github.com/deepanker13))
- Adding parallel support for coveralls ([#1956](https://github.com/kubeflow/training-operator/pull/1956) by [@johnugeorge](https://github.com/johnugeorge))
- chore: pkg import only once ([#1950](https://github.com/kubeflow/training-operator/pull/1950) by [@testwill](https://github.com/testwill))
- fix nproc env in elastic mode for pytorchjob ([#1948](https://github.com/kubeflow/training-operator/pull/1948) by [@kuizhiqing](https://github.com/kuizhiqing))
- Avoid modifying log level globally ([#1944](https://github.com/kubeflow/training-operator/pull/1944) by [@droctothorpe](https://github.com/droctothorpe))
- Add @andreyvelich to Approvers ([#1941](https://github.com/kubeflow/training-operator/pull/1941) by [@andreyvelich](https://github.com/andreyvelich))
- Merge v1.7 branch changes to Main ([#1940](https://github.com/kubeflow/training-operator/pull/1940) by [@johnugeorge](https://github.com/johnugeorge))
- Increase the root volume size on the github runner when building container images ([#1931](https://github.com/kubeflow/training-operator/pull/1931) by [@tenzen-y](https://github.com/tenzen-y))
- Check podGroup CRD for the volcano and the scheudler-plugins as default. ([#1929](https://github.com/kubeflow/training-operator/pull/1929) by [@Syulin7](https://github.com/Syulin7))
- Use a community hosted image in MXJob E2E ([#1928](https://github.com/kubeflow/training-operator/pull/1928) by [@tenzen-y](https://github.com/tenzen-y))
- Build MXJob examples in CI ([#1927](https://github.com/kubeflow/training-operator/pull/1927) by [@tenzen-y](https://github.com/tenzen-y))
- Bump `k8s.io/*` deps to 1.28 ([#1920](https://github.com/kubeflow/training-operator/pull/1920) by [@afritzler](https://github.com/afritzler))
- Replace XGBoost image for E2E with community hosted ([#1922](https://github.com/kubeflow/training-operator/pull/1922) by [@tenzen-y](https://github.com/tenzen-y))
- Creating service account where approriate for MPI Job ([#1917](https://github.com/kubeflow/training-operator/pull/1917) by [@rpemsel](https://github.com/rpemsel))
- Build XGBoostJob example images in CI ([#1913](https://github.com/kubeflow/training-operator/pull/1913) by [@tenzen-y](https://github.com/tenzen-y))
- Manage kube-delivery image from training-operator and update it ([#1909](https://github.com/kubeflow/training-operator/pull/1909) by [@rpemsel](https://github.com/rpemsel))
- Adding Yuki to Approvers ([#1901](https://github.com/kubeflow/training-operator/pull/1901) by [@johnugeorge](https://github.com/johnugeorge))
- docs: Remove reference to tf-operator specific design doc ([#1903](https://github.com/kubeflow/training-operator/pull/1903) by [@terrytangyuan](https://github.com/terrytangyuan))
- Add Training WG Community Call ([#1900](https://github.com/kubeflow/training-operator/pull/1900) by [@andreyvelich](https://github.com/andreyvelich))
- update full change list in changelog ([#1895](https://github.com/kubeflow/training-operator/pull/1895) by [@lowang-bh](https://github.com/lowang-bh))
- update volcano scheduler to 1.8.0 ([#1894](https://github.com/kubeflow/training-operator/pull/1894) by [@lowang-bh](https://github.com/lowang-bh))
- Changelog updated for 1.7.0 rc0 release ([#1892](https://github.com/kubeflow/training-operator/pull/1892) by [@johnugeorge](https://github.com/johnugeorge))
- Add Stale GitHub Action ([#1893](https://github.com/kubeflow/training-operator/pull/1893) by [@andreyvelich](https://github.com/andreyvelich))
- Refactor core/pod tests ([#1890](https://github.com/kubeflow/training-operator/pull/1890) by [@tenzen-y](https://github.com/tenzen-y))
- Remove klog v1 ([#1886](https://github.com/kubeflow/training-operator/pull/1886) by [@tenzen-y](https://github.com/tenzen-y))

[Full Changelog](https://github.com/kubeflow/training-operator/compare/v1.7.0...v1.8.0)

# [v1.8.0-rc.1](https://github.com/kubeflow/training-operator/tree/v1.8.0-rc.1) (2024-06-25)

## Breaking Changes

- [SDK] Support Python 3.11 and Drop Python 3.7 ([#2105](https://github.com/kubeflow/training-operator/pull/2105) by [@tenzen-y](https://github.com/tenzen-y))

## Bug Fixes

- [SDK] Sync Transformers version for train API ([#2147](https://github.com/kubeflow/training-operator/pull/2147) by [@andreyvelich](https://github.com/andreyvelich))
- [SDK] Changed package name to flake8 to fix pip install ([#2140](https://github.com/kubeflow/training-operator/pull/2140) by [@tenzen-y](https://github.com/tenzen-y))
- [SDK] Fix Incorrect Events in get_job_logs API ([#2138](https://github.com/kubeflow/training-operator/pull/2138) by [@tenzen-y](https://github.com/tenzen-y))
- Fix volcano podgroup update issue ([#2079](https://github.com/kubeflow/training-operator/pull/2079) by [@ckyuto](https://github.com/ckyuto))

## Misc

- Refine the integration tests for the immutable PyTorchJob ([#2130](https://github.com/kubeflow/training-operator/pull/2130) by [@tenzen-y](https://github.com/tenzen-y))

[Full Changelog](https://github.com/kubeflow/training-operator/compare/v1.8.0-rc.0...v1.8.0-rc.1)

# [v1.8.0-rc.0](https://github.com/kubeflow/training-operator/tree/v1.8.0-rc.0) (2024-04-28)

## Breaking Changes

- Support K8s v1.29 and Drop K8s v1.26 ([#2039](https://github.com/kubeflow/training-operator/pull/2039) by [@tenzen-y](https://github.com/tenzen-y))
- Support K8s v1.28 and Drop K8s v1.25 ([#2038](https://github.com/kubeflow/training-operator/pull/2038) by [@tenzen-y](https://github.com/tenzen-y))
- Deprecation Notice for MXJob ([#2058](https://github.com/kubeflow/training-operator/pull/2058) by [@tenzen-y](https://github.com/tenzen-y))
- ⚠️ Breaking Changes: Rename `monitoring-port` flag to `webook-server-port` ([#1925](https://github.com/kubeflow/training-operator/pull/1925) by [@afritzler](https://github.com/afritzler))

## New Features

### LLM Fine-Tuning API

- Train/Fine-tune API Proposal for LLMs ([#1945](https://github.com/kubeflow/training-operator/pull/1945) by [@deepanker13](https://github.com/deepanker13))
- [SDK] Train API for LLM Fine-Tuning ([#1962](https://github.com/kubeflow/training-operator/pull/1962) by [@deepanker13](https://github.com/deepanker13))
- Modify LLM Trainer to support BERT and Tiny LLaMA ([#2031](https://github.com/kubeflow/training-operator/pull/2031) by [@andreyvelich](https://github.com/andreyvelich))
- Support arm64 for Hugging Face trainer ([#2028](https://github.com/kubeflow/training-operator/pull/2028) by [@tariq-hasan](https://github.com/tariq-hasan))
- Add Fine-Tune BERT LLM Example ([#2021](https://github.com/kubeflow/training-operator/pull/2021) by [@andreyvelich](https://github.com/andreyvelich))
- Train api dataset download changes ([#1959](https://github.com/kubeflow/training-operator/pull/1959) by [@deepanker13](https://github.com/deepanker13))
- Train api init container creation ([#1958](https://github.com/kubeflow/training-operator/pull/1958) by [@deepanker13](https://github.com/deepanker13))
- [SDK] Add docstring for Train API ([#2075](https://github.com/kubeflow/training-operator/pull/2075) by [@andreyvelich](https://github.com/andreyvelich))

### Control Plane Updates

- Upgrade scheduler-plugins to v0.28.9 ([#2065](https://github.com/kubeflow/training-operator/pull/2065) by [@tenzen-y](https://github.com/tenzen-y))
- Implement webhook validations for the PaddleJob ([#2057](https://github.com/kubeflow/training-operator/pull/2057) by [@tenzen-y](https://github.com/tenzen-y))
- Implement webhook validations for the XGBoostJob ([#2052](https://github.com/kubeflow/training-operator/pull/2052) by [@tenzen-y](https://github.com/tenzen-y))
- Implement webhook validation for the TFJob ([#2051](https://github.com/kubeflow/training-operator/pull/2051) by [@tenzen-y](https://github.com/tenzen-y))
- Implement webhook validations for the PyTorchJob ([#2035](https://github.com/kubeflow/training-operator/pull/2035) by [@tenzen-y](https://github.com/tenzen-y))
- Upgrade PyTorchJob examples to PyTorch v2 ([#2024](https://github.com/kubeflow/training-operator/pull/2024) by [@champon1020](https://github.com/champon1020))
- Upgrade Go version to v1.22 ([#2046](https://github.com/kubeflow/training-operator/pull/2046) by [@tenzen-y](https://github.com/tenzen-y))

### SDK Improvements

- [SDK] Add resources per worker for Create Job API ([#1990](https://github.com/kubeflow/training-operator/pull/1990) by [@andreyvelich](https://github.com/andreyvelich))
- [SDK] Fix Worker and Master templates for PyTorchJob ([#1988](https://github.com/kubeflow/training-operator/pull/1988) by [@andreyvelich](https://github.com/andreyvelich))
- [SDK] Get Kubernetes Events for Job ([#1975](https://github.com/kubeflow/training-operator/pull/1975) by [@andreyvelich](https://github.com/andreyvelich))
- SDK: Upgrade the minimum required Kubernetes version to v1.27.2 ([#2066](https://github.com/kubeflow/training-operator/pull/2066) by [@tenzen-y](https://github.com/tenzen-y))
- [SDK] Add information about TrainingClient logging ([#1973](https://github.com/kubeflow/training-operator/pull/1973) by [@andreyvelich](https://github.com/andreyvelich))
- Training operator SDK unit test ([#1938](https://github.com/kubeflow/training-operator/pull/1938) by [@deepanker13](https://github.com/deepanker13))
- [SDK] Consolidate Naming for CRUD APIs ([#1907](https://github.com/kubeflow/training-operator/pull/1907) by [@andreyvelich](https://github.com/andreyvelich))

## Bug Fixes

- Fix import for HuggingFace Dataset Provider ([#2085](https://github.com/kubeflow/training-operator/pull/2085) by [@andreyvelich](https://github.com/andreyvelich))
- Updated examples for train API ([#2077](https://github.com/kubeflow/training-operator/pull/2077) by [@shruti2522](https://github.com/shruti2522))
- Fail job for non-retryable exit codes ([#2071](https://github.com/kubeflow/training-operator/pull/2071) by [@kellyaa](https://github.com/kellyaa))
- E2E: Replace outdated images with latest ones ([#2083](https://github.com/kubeflow/training-operator/pull/2083) by [@tenzen-y](https://github.com/tenzen-y))
- fix wrong filepath in the simple example command ([#2062](https://github.com/kubeflow/training-operator/pull/2062) by [@qzoscar](https://github.com/qzoscar))
- fix(example): add installation of python-etcd in Pytorch example ([#2064](https://github.com/kubeflow/training-operator/pull/2064) by [@champon1020](https://github.com/champon1020))
- fix: Upgrade controller-gen to v0.14.0 ([#2026](https://github.com/kubeflow/training-operator/pull/2026) by [@champon1020](https://github.com/champon1020))
- Fix build workflow config for pytorch-torchrun-example ([#2020](https://github.com/kubeflow/training-operator/pull/2020) by [@PeterWrighten](https://github.com/PeterWrighten))
- Fix Distributed Data Samplers in PyTorch Examples ([#2012](https://github.com/kubeflow/training-operator/pull/2012) by [@andreyvelich](https://github.com/andreyvelich))
- Fix URL in python SDK setup.py ([#2011](https://github.com/kubeflow/training-operator/pull/2011) by [@garymm](https://github.com/garymm))
- Fix for Github CI to publish HF trainer image ([#1987](https://github.com/kubeflow/training-operator/pull/1987) by [@johnugeorge](https://github.com/johnugeorge))
- train api jupyternotebook fix ([#1984](https://github.com/kubeflow/training-operator/pull/1984) by [@deepanker13](https://github.com/deepanker13))
- fix: volcano podgroup should has a non-empty queue name ([#1977](https://github.com/kubeflow/training-operator/pull/1977) by [@lowang-bh](https://github.com/lowang-bh))
- Fix Master Label for PyTorchJob ([#1974](https://github.com/kubeflow/training-operator/pull/1974) by [@andreyvelich](https://github.com/andreyvelich))
- IsMasterRole fix in pytorchjob controller ([#1969](https://github.com/kubeflow/training-operator/pull/1969) by [@deepanker13](https://github.com/deepanker13))
- [fix] replace ${go env GOPATH} with $(go env GOPATH) to get the prope… ([#1952](https://github.com/kubeflow/training-operator/pull/1952) by [@double12gzh](https://github.com/double12gzh))
- Fixing issues with providing existing service account ([#1918](https://github.com/kubeflow/training-operator/pull/1918) by [@rpemsel](https://github.com/rpemsel))

## Misc

- Update training operator image to latest ([#2089](https://github.com/kubeflow/training-operator/pull/2089) by [@johnugeorge](https://github.com/johnugeorge))
- Update sdk to v1.8.0rc0 ([#2087](https://github.com/kubeflow/training-operator/pull/2087) by [@johnugeorge](https://github.com/johnugeorge))
- Test: Simplify and Identify pod-controller envtest ([#2084](https://github.com/kubeflow/training-operator/pull/2084) by [@tenzen-y](https://github.com/tenzen-y))
- Remove deadcode related to PodDisruptionBudget ([#2073](https://github.com/kubeflow/training-operator/pull/2073) by [@tenzen-y](https://github.com/tenzen-y))
- docs: updating docs for local development ([#2074](https://github.com/kubeflow/training-operator/pull/2074) by [@franciscojavierarceo](https://github.com/franciscojavierarceo))
- PyTorchJob: Always show warnings when using elasticPolicy.nProcPerNode ([#2067](https://github.com/kubeflow/training-operator/pull/2067) by [@tenzen-y](https://github.com/tenzen-y))
- Updated developer docs to include Kind ([#2061](https://github.com/kubeflow/training-operator/pull/2061) by [@franciscojavierarceo](https://github.com/franciscojavierarceo))
- adding fine tune example with s3 as the dataset store ([#2006](https://github.com/kubeflow/training-operator/pull/2006) by [@deepanker13](https://github.com/deepanker13))
- CI: Use a mode=min in the builder cache ([#2053](https://github.com/kubeflow/training-operator/pull/2053) by [@tenzen-y](https://github.com/tenzen-y))
- Fix: upgrade version of crd-ref-docs, which caused panic with go v1.22 ([#2043](https://github.com/kubeflow/training-operator/pull/2043) by [@jdcfd](https://github.com/jdcfd))
- Remove Dockerfile.ppc64le of pytorch example ([#2042](https://github.com/kubeflow/training-operator/pull/2042) by [@champon1020](https://github.com/champon1020))
- publish torchrun example via Dockerfile ([#2018](https://github.com/kubeflow/training-operator/pull/2018) by [@PeterWrighten](https://github.com/PeterWrighten))
- Updated examples/pytorch to disable istio sidecar injection ([#2004](https://github.com/kubeflow/training-operator/pull/2004) by [@jdcfd](https://github.com/jdcfd))
- [docs] development guide update ([#1995](https://github.com/kubeflow/training-operator/pull/1995) by [@shashank-iitbhu](https://github.com/shashank-iitbhu))
- Add Kubeflow Website links to README ([#1983](https://github.com/kubeflow/training-operator/pull/1983) by [@andreyvelich](https://github.com/andreyvelich))
- publish trainer hugging face image ([#1985](https://github.com/kubeflow/training-operator/pull/1985) by [@deepanker13](https://github.com/deepanker13))
- Adding Training image needed for train api ([#1963](https://github.com/kubeflow/training-operator/pull/1963) by [@deepanker13](https://github.com/deepanker13))
- Add test to create PyTorchJob from func ([#1979](https://github.com/kubeflow/training-operator/pull/1979) by [@andreyvelich](https://github.com/andreyvelich))
- Corrected Some Spelling And Grammatical Errors ([#1980](https://github.com/kubeflow/training-operator/pull/1980) by [@daniel-hutao](https://github.com/daniel-hutao))
- torchrun example with cpu version pytorch ([#1965](https://github.com/kubeflow/training-operator/pull/1965) by [@kuizhiqing](https://github.com/kuizhiqing))
- utils changes needed to add train api ([#1954](https://github.com/kubeflow/training-operator/pull/1954) by [@deepanker13](https://github.com/deepanker13))
- Adding parallel support for coveralls ([#1956](https://github.com/kubeflow/training-operator/pull/1956) by [@johnugeorge](https://github.com/johnugeorge))
- chore: pkg import only once ([#1950](https://github.com/kubeflow/training-operator/pull/1950) by [@testwill](https://github.com/testwill))
- fix nproc env in elastic mode for pytorchjob ([#1948](https://github.com/kubeflow/training-operator/pull/1948) by [@kuizhiqing](https://github.com/kuizhiqing))
- Avoid modifying log level globally ([#1944](https://github.com/kubeflow/training-operator/pull/1944) by [@droctothorpe](https://github.com/droctothorpe))
- Add @andreyvelich to Approvers ([#1941](https://github.com/kubeflow/training-operator/pull/1941) by [@andreyvelich](https://github.com/andreyvelich))
- Merge v1.7 branch changes to Main ([#1940](https://github.com/kubeflow/training-operator/pull/1940) by [@johnugeorge](https://github.com/johnugeorge))
- Increase the root volume size on the github runner when building container images ([#1931](https://github.com/kubeflow/training-operator/pull/1931) by [@tenzen-y](https://github.com/tenzen-y))
- Check podGroup CRD for the volcano and the scheudler-plugins as default. ([#1929](https://github.com/kubeflow/training-operator/pull/1929) by [@Syulin7](https://github.com/Syulin7))
- Use a community hosted image in MXJob E2E ([#1928](https://github.com/kubeflow/training-operator/pull/1928) by [@tenzen-y](https://github.com/tenzen-y))
- Build MXJob examples in CI ([#1927](https://github.com/kubeflow/training-operator/pull/1927) by [@tenzen-y](https://github.com/tenzen-y))
- Bump `k8s.io/*` deps to 1.28 ([#1920](https://github.com/kubeflow/training-operator/pull/1920) by [@afritzler](https://github.com/afritzler))
- Replace XGBoost image for E2E with community hosted ([#1922](https://github.com/kubeflow/training-operator/pull/1922) by [@tenzen-y](https://github.com/tenzen-y))
- Creating service account where approriate for MPI Job ([#1917](https://github.com/kubeflow/training-operator/pull/1917) by [@rpemsel](https://github.com/rpemsel))
- Build XGBoostJob example images in CI ([#1913](https://github.com/kubeflow/training-operator/pull/1913) by [@tenzen-y](https://github.com/tenzen-y))
- Manage kube-delivery image from training-operator and update it ([#1909](https://github.com/kubeflow/training-operator/pull/1909) by [@rpemsel](https://github.com/rpemsel))
- Adding Yuki to Approvers ([#1901](https://github.com/kubeflow/training-operator/pull/1901) by [@johnugeorge](https://github.com/johnugeorge))
- docs: Remove reference to tf-operator specific design doc ([#1903](https://github.com/kubeflow/training-operator/pull/1903) by [@terrytangyuan](https://github.com/terrytangyuan))
- Add Training WG Community Call ([#1900](https://github.com/kubeflow/training-operator/pull/1900) by [@andreyvelich](https://github.com/andreyvelich))
- update full change list in changelog ([#1895](https://github.com/kubeflow/training-operator/pull/1895) by [@lowang-bh](https://github.com/lowang-bh))
- update volcano scheduler to 1.8.0 ([#1894](https://github.com/kubeflow/training-operator/pull/1894) by [@lowang-bh](https://github.com/lowang-bh))
- Changelog updated for 1.7.0 rc0 release ([#1892](https://github.com/kubeflow/training-operator/pull/1892) by [@johnugeorge](https://github.com/johnugeorge))
- Add Stale GitHub Action ([#1893](https://github.com/kubeflow/training-operator/pull/1893) by [@andreyvelich](https://github.com/andreyvelich))
- Refactor core/pod tests ([#1890](https://github.com/kubeflow/training-operator/pull/1890) by [@tenzen-y](https://github.com/tenzen-y))
- Remove klog v1 ([#1886](https://github.com/kubeflow/training-operator/pull/1886) by [@tenzen-y](https://github.com/tenzen-y))

[Full Changelog](https://github.com/kubeflow/training-operator/compare/v1.7.0...v1.8.0-rc.0)

# [v1.7.0-rc.0](https://github.com/kubeflow/training-operator/tree/v1.7.0-rc.0) (2023-07-07)

[Full Changelog](https://github.com/kubeflow/training-operator/compare/v1.6.0...v1.7.0-rc.0)

## Breaking Changes

- Upgrade Scheduler Plugins version to v0.25.7 https://github.com/kubeflow/training-operator/pull/1824 ([tenzen-y](https://github.com/tenzen-y))
- Upgrade the kubernetes dependencies to v1.27 https://github.com/kubeflow/training-operator/pull/1834 ([tenzen-y](https://github.com/tenzen-y))

## New Features

- Make scheduler-plugins the default gang scheduler. [\#1747](https://github.com/kubeflow/training-operator/pull/1747) ([Syulin7](https://github.com/Syulin7))
- Merge kubeflow/common to training-operator [\#1813](https://github.com/kubeflow/training-operator/pull/1813) ([johnugeorge](https://github.com/johnugeorge))
- Auto-generate RBAC manifests by the controller-gen [\#1815](https://github.com/kubeflow/training-operator/pull/1815) ([Syulin7](https://github.com/Syulin7))
- Implement suspend semantics [\#1859](https://github.com/kubeflow/training-operator/pull/1859) ([tenzen-y](https://github.com/tenzen-y))
- Set up controllers using goroutines to start the manager quickly [\#1869](https://github.com/kubeflow/training-operator/pull/1869) ([tenzen-y](https://github.com/tenzen-y))
- Set correct ENV for PytorchJob to support torchrun [\#1840](https://github.com/kubeflow/training-operator/pull/1840) ([kuizhiqing](https://github.com/kuizhiqing))

## Bug Fixes

- Fix a bug that XGBoostJob's running condition isn't updated when the job is resumed [\#1866](https://github.com/kubeflow/training-operator/pull/1866) ([tenzen-y](https://github.com/tenzen-y))
- Set a Running condition when the XGBoostJob is completed and doesn't have a Running condition [\#1789](https://github.com/kubeflow/training-operator/pull/1789) ([tenzen-y](https://github.com/tenzen-y))
- Avoid to depend on local env when installing the code-generators [\#1810](https://github.com/kubeflow/training-operator/pull/1810) ([tenzen-y](https://github.com/tenzen-y))

## Misc

- Removing reconciler code [\#1879](https://github.com/kubeflow/training-operator/pull/1879) ([johnugeorge](https://github.com/johnugeorge))
- Make Condition and ReplicaStatus optional [\#1862](https://github.com/kubeflow/training-operator/pull/1862) ([tenzen-y](https://github.com/tenzen-y))
- Use the same reasons for Condition and Event [\#1854](https://github.com/kubeflow/training-operator/pull/1854) ([tenzen-y](https://github.com/tenzen-y))
- Fully consolidate tfjob-operator to training-operator [\#1850](https://github.com/kubeflow/training-operator/pull/1850) ([tenzen-y](https://github.com/tenzen-y))
- Clean up /pkg/common/util/v1 [\#1845](https://github.com/kubeflow/training-operator/pull/1845) ([tenzen-y](https://github.com/tenzen-y))
- Refactoring tests in common/controller.v1 [\#1843](https://github.com/kubeflow/training-operator/pull/1843) ([tenzen-y](https://github.com/tenzen-y))
- remove duplicate code of add task spec annotation [\#1839](https://github.com/kubeflow/training-operator/pull/1839) ([lowang-bh](https://github.com/lowang-bh))
- fetch volcano log when e2e failed [\#1837](https://github.com/kubeflow/training-operator/pull/1837) ([lowang-bh](https://github.com/lowang-bh))
- Add check pods are not scheduled when testing gang-scheduler integrations in e2e [\#1835](https://github.com/kubeflow/training-operator/pull/1835) ([tenzen-y](https://github.com/tenzen-y))
- Replace dummy client with fake client [\#1818](https://github.com/kubeflow/training-operator/pull/1818) ([tenzen-y](https://github.com/tenzen-y))
- Add default Intel MPI env variables to MPIJob [\#1804](https://github.com/kubeflow/training-operator/pull/1804) ([tkatila](https://github.com/tkatila))
- Improve E2E tests for the gang-scheduling [\#1801](https://github.com/kubeflow/training-operator/pull/1801) ([tenzen-y](https://github.com/tenzen-y))
- xgb yaml container name should be consistent with xgb job default container name [\#1794](https://github.com/kubeflow/training-operator/pull/1794) ([Crisescode](https://github.com/Crisescode))
- make timeout configurable from e2e tests [\#1787](https://github.com/kubeflow/training-operator/pull/1787) ([nagar-ajay](https://github.com/nagar-ajay))

# [v1.6.0](https://github.com/kubeflow/training-operator/tree/v1.6.0) (2023-03-21)

Note: Since scheduler-plugins has changed API from `sigs.k8s.io` with the `x-k8s.io`, future releases of training operator(v1.7+) will not support scheduler-plugins v0.24.x or lower. Related: [\#1769](https://github.com/kubeflow/training-operator/issues/1769)

Note: Latest [Python SDK 1.6 version](https://pypi.org/project/kubeflow-training/1.6.0/) does not support earlier training operator versions. The minimum training operator version required is v1.6.0 release. Related: [\#1702](https://github.com/kubeflow/training-operator/pull/1702)

[Full Changelog](https://github.com/kubeflow/training-operator/compare/v1.5.0...v1.6.0)

## New Features

- Support for k8s v1.25 in CI [\#1684](https://github.com/kubeflow/training-operator/pull/1684) ([johnugeorge](https://github.com/johnugeorge))
- HPA support for PyTorch Elastic [\#1701](https://github.com/kubeflow/training-operator/pull/1701) ([johnugeorge](https://github.com/johnugeorge))
- Adopting coschduling plugin [\#1724](https://github.com/kubeflow/training-operator/pull/1724) ([tenzen-y](https://github.com/tenzen-y))
- Support for Paddlepaddle [\#1675](https://github.com/kubeflow/training-operator/pull/1675) ([kuizhiqing](https://github.com/kuizhiqing))
- Create TFJob and PyTorchJob from Function APIs in the Training SDK [\#1659](https://github.com/kubeflow/training-operator/pull/1659) ([andreyvelich](https://github.com/andreyvelich))
- \[SDK\] Use Training Client without Kube Config [\#1740](https://github.com/kubeflow/training-operator/pull/1740) ([andreyvelich](https://github.com/andreyvelich))
- \[SDK\] Create Unify Training Client [\#1719](https://github.com/kubeflow/training-operator/pull/1719) ([andreyvelich](https://github.com/andreyvelich))

## Bug Fixes

- [SDK] pod has no metadata attr anymore in the get_job_logs\(\) … [\#1760](https://github.com/kubeflow/training-operator/pull/1760) ([yaobaiwei](https://github.com/yaobaiwei))
- Add PodGroup as controller watch source [\#1666](https://github.com/kubeflow/training-operator/pull/1666) ([ggaaooppeenngg](https://github.com/ggaaooppeenngg))
- fix infinite loop in init-pytorch container [\#1756](https://github.com/kubeflow/training-operator/pull/1756) ([kidddddddddddddddddddddd](https://github.com/kidddddddddddddddddddddd))
- Fix the success condition of the job in PyTorchJob's Elastic mode. [\#1752](https://github.com/kubeflow/training-operator/pull/1752) ([Syulin7](https://github.com/Syulin7))
- Fix XGBoost conditions bug [\#1737](https://github.com/kubeflow/training-operator/pull/1737) ([tenzen-y](https://github.com/tenzen-y))
- To fix scaledown error, upgrade PyTorch version to v1.13.1 in echo example [\#1733](https://github.com/kubeflow/training-operator/pull/1733) ([tenzen-y](https://github.com/tenzen-y))
- fix: support MxNet single host training when update mxJob status [\#1644](https://github.com/kubeflow/training-operator/pull/1644) ([PeterChg](https://github.com/PeterChg))
- fix: fix mxnet failed to update StartTime and CompletionTime [\#1643](https://github.com/kubeflow/training-operator/pull/1643) ([PeterChg](https://github.com/PeterChg))
- Fix the default LeaderElectionID and make it an argument [\#1639](https://github.com/kubeflow/training-operator/pull/1639) ([goyalankit](https://github.com/goyalankit))
- fix: fix wrong parameter for resolveControllerRef [\#1583](https://github.com/kubeflow/training-operator/pull/1583) ([fighterhit](https://github.com/fighterhit))
- fix: tfjob with restartPolicy=ExitCode not work [\#1562](https://github.com/kubeflow/training-operator/pull/1562) ([cheimu](https://github.com/cheimu))
- fix: Mac M1 compatible Dockerfile and bump TF version [\#1700](https://github.com/kubeflow/training-operator/pull/1700) ([terrytangyuan](https://github.com/terrytangyuan))
- Fix status lost [\#1697](https://github.com/kubeflow/training-operator/pull/1697) ([ggaaooppeenngg](https://github.com/ggaaooppeenngg))
- handle all restart policies [\#1649](https://github.com/kubeflow/training-operator/pull/1649) ([abin-thomas-by](https://github.com/abin-thomas-by))
- \[chore\] fix typo [\#1648](https://github.com/kubeflow/training-operator/pull/1648) ([tenzen-y](https://github.com/tenzen-y))

## Misc

- Add validation for verifying that the CustomJob \(e.g., TFJob\) name meets DNS1035 [\#1748](https://github.com/kubeflow/training-operator/pull/1748) ([tenzen-y](https://github.com/tenzen-y))
- Configure controller worker threads [\#1707](https://github.com/kubeflow/training-operator/pull/1707) ([HeGaoYuan](https://github.com/HeGaoYuan))
- Validation Spec consistency [\#1705](https://github.com/kubeflow/training-operator/pull/1705) ([HeGaoYuan](https://github.com/HeGaoYuan))
- \[SDK\] Remove Final Keyword from constants [\#1676](https://github.com/kubeflow/training-operator/pull/1676) ([andreyvelich](https://github.com/andreyvelich))
- Fix Python installation in CI [\#1759](https://github.com/kubeflow/training-operator/pull/1759) ([tenzen-y](https://github.com/tenzen-y))
- Update mpijob_controller.go [\#1755](https://github.com/kubeflow/training-operator/pull/1755) ([yshalabi](https://github.com/yshalabi))
- Set the default value of CleanPodPolicy to None [\#1754](https://github.com/kubeflow/training-operator/pull/1754) ([Syulin7](https://github.com/Syulin7))
- Update join Slack link [\#1750](https://github.com/kubeflow/training-operator/pull/1750) ([Syulin7](https://github.com/Syulin7))
- Update latest operator image [\#1742](https://github.com/kubeflow/training-operator/pull/1742) ([johnugeorge](https://github.com/johnugeorge))
- Run E2E with various Python versions to verify Python SDK [\#1741](https://github.com/kubeflow/training-operator/pull/1741) ([tenzen-y](https://github.com/tenzen-y))
- Add Yuki to reviewer group [\#1739](https://github.com/kubeflow/training-operator/pull/1739) ([johnugeorge](https://github.com/johnugeorge))
- Trim down CRD descriptions [\#1735](https://github.com/kubeflow/training-operator/pull/1735) ([tenzen-y](https://github.com/tenzen-y))
- Add CI to build example images [\#1731](https://github.com/kubeflow/training-operator/pull/1731) ([tenzen-y](https://github.com/tenzen-y))
- Fix predicates of paddlepaddle-controller for scheduling.volcano.sh/v1beta1 PodGroup [\#1730](https://github.com/kubeflow/training-operator/pull/1730) ([tenzen-y](https://github.com/tenzen-y))
- Fix indents on examples for tensorflow [\#1726](https://github.com/kubeflow/training-operator/pull/1726) ([tenzen-y](https://github.com/tenzen-y))
- docs: Update Kubernetes requirement and version matrix [\#1721](https://github.com/kubeflow/training-operator/pull/1721) ([terrytangyuan](https://github.com/terrytangyuan))
- chore: Update the use of MultiWorkerMirroredStrategy in TF [\#1715](https://github.com/kubeflow/training-operator/pull/1715) ([terrytangyuan](https://github.com/terrytangyuan))
- Removing deprecated Job Labels [\#1702](https://github.com/kubeflow/training-operator/pull/1702) ([johnugeorge](https://github.com/johnugeorge))
- Bump certifi from 2022.9.14 to 2022.12.7 in /py/kubeflow/tf_operator [\#1699](https://github.com/kubeflow/training-operator/pull/1699) ([dependabot[bot]](https://github.com/apps/dependabot))
- Add myself to reviewer. [\#1689](https://github.com/kubeflow/training-operator/pull/1689) ([kuizhiqing](https://github.com/kuizhiqing))
- Upgrade the envtest version [\#1687](https://github.com/kubeflow/training-operator/pull/1687) ([tenzen-y](https://github.com/tenzen-y))
- \[chore\] Upgrade some actions version [\#1686](https://github.com/kubeflow/training-operator/pull/1686) ([tenzen-y](https://github.com/tenzen-y))
- Upgrade Golangci-lint [\#1685](https://github.com/kubeflow/training-operator/pull/1685) ([johnugeorge](https://github.com/johnugeorge))
- Make a generic logger instead of the nil logger on dependent update [\#1680](https://github.com/kubeflow/training-operator/pull/1680) ([ggaaooppeenngg](https://github.com/ggaaooppeenngg))
- Bump protobuf from 3.8.0 to 3.18.3 in /py/kubeflow/tf_operator [\#1669](https://github.com/kubeflow/training-operator/pull/1669) ([dependabot[bot]](https://github.com/apps/dependabot))
- Removed GOARCH dependency for multiarch support [\#1674](https://github.com/kubeflow/training-operator/pull/1674) ([pranavpandit1](https://github.com/pranavpandit1))
- Update deployment.yaml [\#1668](https://github.com/kubeflow/training-operator/pull/1668) ([OmriShiv](https://github.com/OmriShiv))
- Upgrade Go version to v1.19 [\#1663](https://github.com/kubeflow/training-operator/pull/1663) ([tenzen-y](https://github.com/tenzen-y))
- Upgrade kubernetes versoin for test [\#1667](https://github.com/kubeflow/training-operator/pull/1667) ([tenzen-y](https://github.com/tenzen-y))
- Adding support for linux/ppc64le in github actions for training-operator [\#1692](https://github.com/kubeflow/training-operator/pull/1692) ([amitmukati-2604](https://github.com/amitmukati-2604))
- style: Refine name and signature of 2 replicaName functions [\#1660](https://github.com/kubeflow/training-operator/pull/1660) ([houz42](https://github.com/houz42))
- Update training operator sdk version to 1.5.0 [\#1651](https://github.com/kubeflow/training-operator/pull/1651) ([johnugeorge](https://github.com/johnugeorge))
- Add finalizers to cluster-role [\#1646](https://github.com/kubeflow/training-operator/pull/1646) ([ArangoGutierrez](https://github.com/ArangoGutierrez))
- Update the cmd to support MPI operator in ReadME [\#1656](https://github.com/kubeflow/training-operator/pull/1656) ([denkensk](https://github.com/denkensk))

## Closed issues

- The default value for CleanPodPolicy is inconsistent. [\#1753](https://github.com/kubeflow/training-operator/issues/1753)
- HPA support for PyTorch Elastic [\#1751](https://github.com/kubeflow/training-operator/issues/1751)
- Bug: allowance of non DNS-1035 compliant PyTorchJob names results in service creation failures and missing state [\#1745](https://github.com/kubeflow/training-operator/issues/1745)
- paddle-operator can not get podgroup status\(inqueue\) with volcano when enable gang [\#1729](https://github.com/kubeflow/training-operator/issues/1729)
- \*job API\(master\) cannot compatible with old job [\#1725](https://github.com/kubeflow/training-operator/issues/1725)
- Support coscheduling plugin [\#1722](https://github.com/kubeflow/training-operator/issues/1722)
- Number of worker threads used by the controller can't be configured [\#1706](https://github.com/kubeflow/training-operator/issues/1706)
- Conformance: Training tests [\#1698](https://github.com/kubeflow/training-operator/issues/1698)
- PyTorch and MPI Operator pulls hardcoded initContainer [\#1696](https://github.com/kubeflow/training-operator/issues/1696)
- PaddlePaddle Training: why can't find pods [\#1694](https://github.com/kubeflow/training-operator/issues/1694)
- Training-operator pod CrashLoopBackOff in K8s v1.23.6 with kubeflow1.6.1 [\#1693](https://github.com/kubeflow/training-operator/issues/1693)
- \[SDK\] Create unify client for all Training Job types [\#1691](https://github.com/kubeflow/training-operator/issues/1691)
- Support Kubernetes v1.25 [\#1682](https://github.com/kubeflow/training-operator/issues/1682)
- panic happened when add podgroup watch [\#1679](https://github.com/kubeflow/training-operator/issues/1679)
- OnDependentUpdateFunc for Job will panic when enable volcano scheduler [\#1678](https://github.com/kubeflow/training-operator/issues/1678)
- There is no clusterrole of "MPI Jobs" in kubeflow 1.5. [\#1670](https://github.com/kubeflow/training-operator/issues/1670)
- Change Kubernetes version for test [\#1665](https://github.com/kubeflow/training-operator/issues/1665)
- Support for multiplatform container imege \(amd64 and arm64\) [\#1664](https://github.com/kubeflow/training-operator/issues/1664)
- Training Operator pod failed to start on OCP 4.10.30 with error "memory limit too low" [\#1661](https://github.com/kubeflow/training-operator/issues/1661)
- After setting hostNetwork to true, mpi does not work [\#1657](https://github.com/kubeflow/training-operator/issues/1657)
- What is the purpose of /examples/pytorch/elastic/etcd.yaml [\#1655](https://github.com/kubeflow/training-operator/issues/1655)
- When will MPIJob support v2beta1 version? [\#1653](https://github.com/kubeflow/training-operator/issues/1653)
- Kubernetes HPA doesn't work with elastic PytorchJob [\#1645](https://github.com/kubeflow/training-operator/issues/1645)
- training-operator can not get podgroup status\(inqueue\) with volcano when enable gang [\#1630](https://github.com/kubeflow/training-operator/issues/1630)
- Training operator fails to create HPA for TorchElastic jobs [\#1626](https://github.com/kubeflow/training-operator/issues/1626)
- Release v1.5.0 tracking [\#1622](https://github.com/kubeflow/training-operator/issues/1622)
- upgrade client-go [\#1599](https://github.com/kubeflow/training-operator/issues/1599)
- trainning-operator may need to monitor PodGroup [\#1574](https://github.com/kubeflow/training-operator/issues/1574)
- Error: invalid memory address or nil pointer dereference [\#1553](https://github.com/kubeflow/training-operator/issues/1553)
- The pytorchJob training is slow [\#1532](https://github.com/kubeflow/training-operator/issues/1532)
- pytorch elastic scheduler error [\#1504](https://github.com/kubeflow/training-operator/issues/1504)

# [v1.4.0-rc.0](https://github.com/kubeflow/training-operator/tree/v1.4.0-rc.0) (2022-01-26)

[Full Changelog](https://github.com/kubeflow/training-operator/compare/v1.3.0...v1.4.0-rc.0)

## Features and Improvements

- Display coverage % in GitHub actions list [\#1442](https://github.com/kubeflow/training-operator/issues/1442)
- Add Go test to CI [\#1436](https://github.com/kubeflow/training-operator/issues/1436)

## Fixed Bugs

- \[bug\] Missing init container in PyTorchJob [\#1482](https://github.com/kubeflow/training-operator/issues/1482)
- Fail to install tf-operator in minikube because of the version of kubectl/kustomize [\#1381](https://github.com/kubeflow/training-operator/issues/1381)

## Closed Issues

- Restore KUBEFLOW_NAMESPACE options [\#1522](https://github.com/kubeflow/training-operator/issues/1522)
- Improve test coverage [\#1497](https://github.com/kubeflow/training-operator/issues/1497)
- swagger.json missing Pytorchjob.Spec.ElasticPolicy [\#1483](https://github.com/kubeflow/training-operator/issues/1483)
- PytorchJob DDP training will stop if I delete a worker pod [\#1478](https://github.com/kubeflow/training-operator/issues/1478)
- Write down e2e failure debug process [\#1467](https://github.com/kubeflow/training-operator/issues/1467)
- How can i add the Priorityclass to the TFjob？ [\#1466](https://github.com/kubeflow/training-operator/issues/1466)
- github.com/go-logr/zapr.\(\*zapLogger\).Error [\#1444](https://github.com/kubeflow/training-operator/issues/1444)
- Podgroup is constantly created and deleted after tfjob is success or failure [\#1426](https://github.com/kubeflow/training-operator/issues/1426)
- Cut official release of 1.3.0 [\#1425](https://github.com/kubeflow/training-operator/issues/1425)
- Add "not maintained" notice to other operator repos [\#1423](https://github.com/kubeflow/training-operator/issues/1423)
- Python SDK for Kubeflow Training Operator [\#1380](https://github.com/kubeflow/training-operator/issues/1380)

## Merged Pull Requests

- Update manifests with latest image tag [\#1527](https://github.com/kubeflow/training-operator/pull/1527) ([johnugeorge](https://github.com/johnugeorge))
- add option for mpi kubectl delivery [\#1525](https://github.com/kubeflow/training-operator/pull/1525) ([zw0610](https://github.com/zw0610))
- restore option namespace in launch arguments [\#1524](https://github.com/kubeflow/training-operator/pull/1524) ([zw0610](https://github.com/zw0610))
- remove unused scripts [\#1521](https://github.com/kubeflow/training-operator/pull/1521) ([zw0610](https://github.com/zw0610))
- remove ChanYiLin from approvers [\#1513](https://github.com/kubeflow/training-operator/pull/1513) ([ChanYiLin](https://github.com/ChanYiLin))
- add StacktraceLevel for zapr [\#1512](https://github.com/kubeflow/training-operator/pull/1512) ([qiankunli](https://github.com/qiankunli))
- add unit tests for tensorflow controller [\#1511](https://github.com/kubeflow/training-operator/pull/1511) ([zw0610](https://github.com/zw0610))
- add the example of MPIJob [\#1508](https://github.com/kubeflow/training-operator/pull/1508) ([hackerboy01](https://github.com/hackerboy01))
- Added 2022 roadmap and migrated previous roadmap from kubeflow/common [\#1500](https://github.com/kubeflow/training-operator/pull/1500) ([terrytangyuan](https://github.com/terrytangyuan))
- Fix a typo in mpi controller log [\#1495](https://github.com/kubeflow/training-operator/pull/1495) ([LuBingtan](https://github.com/LuBingtan))
- feat\(pytorch\): Add init container config to avoid DNS lookup failure [\#1493](https://github.com/kubeflow/training-operator/pull/1493) ([gaocegege](https://github.com/gaocegege))
- chore: Fix GitHub Actions script [\#1491](https://github.com/kubeflow/training-operator/pull/1491) ([tenzen-y](https://github.com/tenzen-y))
- chore: Fix missspell in tfjob [\#1490](https://github.com/kubeflow/training-operator/pull/1490) ([tenzen-y](https://github.com/tenzen-y))
- chore: Update OWNERS [\#1489](https://github.com/kubeflow/training-operator/pull/1489) ([gaocegege](https://github.com/gaocegege))
- Bump jinja2 from 2.10.1 to 2.11.3 in /py/kubeflow/tf_operator [\#1487](https://github.com/kubeflow/training-operator/pull/1487) ([dependabot[bot]](https://github.com/apps/dependabot))
- fix comments for mpi-controller [\#1485](https://github.com/kubeflow/training-operator/pull/1485) ([hackerboy01](https://github.com/hackerboy01))
- add expectation-related functions for other resources used in mpi-controller [\#1484](https://github.com/kubeflow/training-operator/pull/1484) ([zw0610](https://github.com/zw0610))
- Add MPI job to README now that it's supported [\#1480](https://github.com/kubeflow/training-operator/pull/1480) ([terrytangyuan](https://github.com/terrytangyuan))
- add mpi doc [\#1477](https://github.com/kubeflow/training-operator/pull/1477) ([zw0610](https://github.com/zw0610))
- Set Go version of base image to 1.17 [\#1476](https://github.com/kubeflow/training-operator/pull/1476) ([tenzen-y](https://github.com/tenzen-y))
- update label for tf-controller [\#1474](https://github.com/kubeflow/training-operator/pull/1474) ([zw0610](https://github.com/zw0610))
- Add Akuity to the list of adopters [\#1473](https://github.com/kubeflow/training-operator/pull/1473) ([terrytangyuan](https://github.com/terrytangyuan))
- Add PR template with doc checklist [\#1470](https://github.com/kubeflow/training-operator/pull/1470) ([andreyvelich](https://github.com/andreyvelich))
- Add e2e failure debugging guidance [\#1469](https://github.com/kubeflow/training-operator/pull/1469) ([Jeffwan](https://github.com/Jeffwan))
- chore: Add .gitattributes to ignore Jsonnet test code for linguist [\#1463](https://github.com/kubeflow/training-operator/pull/1463) ([terrytangyuan](https://github.com/terrytangyuan))
- Migrate additional examples from xgboost-operator [\#1461](https://github.com/kubeflow/training-operator/pull/1461) ([terrytangyuan](https://github.com/terrytangyuan))
- Minor edits to README.md [\#1460](https://github.com/kubeflow/training-operator/pull/1460) ([terrytangyuan](https://github.com/terrytangyuan))
- add mpi-operator\(v1\) to the unified operator [\#1457](https://github.com/kubeflow/training-operator/pull/1457) ([hackerboy01](https://github.com/hackerboy01))
- fix tfjob status when enableDynamicWorker set true [\#1455](https://github.com/kubeflow/training-operator/pull/1455) ([zw0610](https://github.com/zw0610))
- feat\(pytorch\): Support elastic training [\#1453](https://github.com/kubeflow/training-operator/pull/1453) ([gaocegege](https://github.com/gaocegege))
- fix: generate printer columns for job crds [\#1451](https://github.com/kubeflow/training-operator/pull/1451) ([henrysecond1](https://github.com/henrysecond1))
- Fix README typo [\#1450](https://github.com/kubeflow/training-operator/pull/1450) ([davidxia](https://github.com/davidxia))
- consistent naming for better readability [\#1449](https://github.com/kubeflow/training-operator/pull/1449) ([pramodrj07](https://github.com/pramodrj07))
- Fix set scheduler error [\#1448](https://github.com/kubeflow/training-operator/pull/1448) ([qiankunli](https://github.com/qiankunli))
- Add CI to run the tests for Go [\#1440](https://github.com/kubeflow/training-operator/pull/1440) ([tenzen-y](https://github.com/tenzen-y))
- fix: Add missing retrying package that failed the import [\#1439](https://github.com/kubeflow/training-operator/pull/1439) ([terrytangyuan](https://github.com/terrytangyuan))
- Generate a single `swagger.json` file for all frameworks [\#1437](https://github.com/kubeflow/training-operator/pull/1437) ([alembiewski](https://github.com/alembiewski))
- Update links and files with the new URL [\#1434](https://github.com/kubeflow/training-operator/pull/1434) ([andreyvelich](https://github.com/andreyvelich))
- chore: update CHANGELOG.md [\#1432](https://github.com/kubeflow/training-operator/pull/1432) ([Jeffwan](https://github.com/Jeffwan))
- Add acknowledgement section in README to credit all contributors [\#1422](https://github.com/kubeflow/training-operator/pull/1422) ([terrytangyuan](https://github.com/terrytangyuan))
- Add Cisco to Adopters List [\#1421](https://github.com/kubeflow/training-operator/pull/1421) ([andreyvelich](https://github.com/andreyvelich))
- Add Python SDK for Kubeflow Training Operator [\#1420](https://github.com/kubeflow/training-operator/pull/1420) ([alembiewski](https://github.com/alembiewski))
- docs: Move myself to approvers [\#1419](https://github.com/kubeflow/training-operator/pull/1419) ([terrytangyuan](https://github.com/terrytangyuan))
- fix hyperlinks in the 'overview' section [\#1418](https://github.com/kubeflow/training-operator/pull/1418) ([pramodrj07](https://github.com/pramodrj07))
- docs: Migrate adopters of all operators to this repo [\#1417](https://github.com/kubeflow/training-operator/pull/1417) ([terrytangyuan](https://github.com/terrytangyuan))
- Feature/support pytorchjob set queue of volcano [\#1415](https://github.com/kubeflow/training-operator/pull/1415) ([qiankunli](https://github.com/qiankunli))
- Bump controller-tools to 0.6.0 and enable GenerateEmbeddedObjectMeta [\#1409](https://github.com/kubeflow/training-operator/pull/1409) ([Jeffwan](https://github.com/Jeffwan))
- Update scripts to generate sdk for all frameworks [\#1389](https://github.com/kubeflow/training-operator/pull/1389) ([Jeffwan](https://github.com/Jeffwan))

# [v1.3.0](https://github.com/kubeflow/training-operator/tree/v1.3.0) (2021-10-03)

[Full Changelog](https://github.com/kubeflow/training-operator/compare/v1.3.0-rc.2...v1.3.0)

## Fixed Bugs

- Unable to specify pod template metadata for TFJob [\#1403](https://github.com/kubeflow/training-operator/issues/1403)

# [v1.3.0-rc.2](https://github.com/kubeflow/training-operator/tree/v1.3.0-rc.2) (2021-09-21)

[Full Changelog](https://github.com/kubeflow/training-operator/compare/v1.3.0-rc.1...v1.3.0-rc.2)

## Fixed Bugs

- Missing Pod label for Service selector [\#1399](https://github.com/kubeflow/training-operator/issues/1399)

# [v1.3.0-rc.1](https://github.com/kubeflow/training-operator/tree/v1.3.0-rc.1) (2021-09-15)

[Full Changelog](https://github.com/kubeflow/training-operator/compare/v1.3.0-rc.0...v1.3.0-rc.1)

## Fixed Bugs

- \[bug\] Reconcilation fails when upgrading common to 0.3.6 [\#1394](https://github.com/kubeflow/training-operator/issues/1394)

## Merged Pull Requests

- Update manifests with latest image tag [\#1406](https://github.com/kubeflow/training-operator/pull/1406) ([johnugeorge](https://github.com/johnugeorge))
- 2010: fix to expose correct monitoring port [\#1405](https://github.com/kubeflow/training-operator/pull/1405) ([deepak-muley](https://github.com/deepak-muley))
- Fix 1399: added pod matching label in service selector [\#1404](https://github.com/kubeflow/training-operator/pull/1404) ([deepak-muley](https://github.com/deepak-muley))
- fix: runPolicy validation error in the examples [\#1401](https://github.com/kubeflow/training-operator/pull/1401) ([Jeffwan](https://github.com/Jeffwan))

# [v1.3.0-rc.0](https://github.com/kubeflow/training-operator/tree/v1.3.0-rc.0) (2021-08-31)

[Full Changelog](https://github.com/kubeflow/training-operator/compare/v1.3.0-alpha.3...v1.3.0-rc.0)

## Merged Pull Requests

- chore: Update training-operator tag [\#1396](https://github.com/kubeflow/training-operator/pull/1396) ([Jeffwan](https://github.com/Jeffwan))
- Add simple verification jobs [\#1391](https://github.com/kubeflow/training-operator/pull/1391) ([Jeffwan](https://github.com/Jeffwan))
- fix: volcano pod group creation issue [\#1390](https://github.com/kubeflow/training-operator/pull/1390) ([Jeffwan](https://github.com/Jeffwan))
- chore: Bump kubeflow/common version to 0.3.7 [\#1388](https://github.com/kubeflow/training-operator/pull/1388) ([Jeffwan](https://github.com/Jeffwan))

# [v1.3.0-alpha.3](https://github.com/kubeflow/training-operator/tree/v1.3.0-alpha.3) (2021-08-29)

[Full Changelog](https://github.com/kubeflow/training-operator/compare/v1.2.1...v1.3.0-alpha.3)

## Closed Issues

- Update guidance to install all-in-one operator in README.md [\#1386](https://github.com/kubeflow/training-operator/issues/1386)

## Merged Pull Requests

- chore\(doc\): Update README.md [\#1387](https://github.com/kubeflow/training-operator/pull/1387) ([Jeffwan](https://github.com/Jeffwan))
- Remove tf-operator from the codebase [\#1378](https://github.com/kubeflow/training-operator/pull/1378) ([thunderboltsid](https://github.com/thunderboltsid))

# [v1.2.1](https://github.com/kubeflow/training-operator/tree/v1.2.1) (2021-08-27)

[Full Changelog](https://github.com/kubeflow/training-operator/compare/v1.3.0-alpha.2...v1.2.1)

# [v1.3.0-alpha.2](https://github.com/kubeflow/training-operator/tree/v1.3.0-alpha.2) (2021-08-15)

[Full Changelog](https://github.com/kubeflow/training-operator/compare/v1.3.0-alpha.1...v1.3.0-alpha.2)

# [v1.3.0-alpha.1](https://github.com/kubeflow/training-operator/tree/v1.3.0-alpha.1) (2021-08-13)

[Full Changelog](https://github.com/kubeflow/training-operator/compare/v1.2.0...v1.3.0-alpha.1)

# [v1.2.0](https://github.com/kubeflow/training-operator/tree/v1.2.0) (2021-08-03)

[Full Changelog](https://github.com/kubeflow/training-operator/compare/v1.1.0...v1.2.0)

# [v1.1.0](https://github.com/kubeflow/training-operator/tree/v1.1.0) (2021-03-20)

[Full Changelog](https://github.com/kubeflow/training-operator/compare/v1.0.1-rc.5...v1.1.0)

# [v1.0.1-rc.5](https://github.com/kubeflow/training-operator/tree/v1.0.1-rc.5) (2021-02-09)

[Full Changelog](https://github.com/kubeflow/training-operator/compare/v1.0.1-rc.4...v1.0.1-rc.5)

# [v1.0.1-rc.4](https://github.com/kubeflow/training-operator/tree/v1.0.1-rc.4) (2021-02-04)

[Full Changelog](https://github.com/kubeflow/training-operator/compare/v1.0.1-rc.3...v1.0.1-rc.4)

# [v1.0.1-rc.3](https://github.com/kubeflow/training-operator/tree/v1.0.1-rc.3) (2021-01-27)

[Full Changelog](https://github.com/kubeflow/training-operator/compare/v1.0.1-rc.2...v1.0.1-rc.3)

# [v1.0.1-rc.2](https://github.com/kubeflow/training-operator/tree/v1.0.1-rc.2) (2021-01-27)

[Full Changelog](https://github.com/kubeflow/training-operator/compare/v1.0.1-rc.1...v1.0.1-rc.2)

# [v1.0.1-rc.1](https://github.com/kubeflow/training-operator/tree/v1.0.1-rc.1) (2021-01-18)

[Full Changelog](https://github.com/kubeflow/training-operator/compare/v1.0.1-rc.0...v1.0.1-rc.1)

# [v1.0.1-rc.0](https://github.com/kubeflow/training-operator/tree/v1.0.1-rc.0) (2020-12-22)

[Full Changelog](https://github.com/kubeflow/training-operator/compare/v1.0.0-rc.0...v1.0.1-rc.0)

# [v1.0.0-rc.0](https://github.com/kubeflow/training-operator/tree/v1.0.0-rc.0) (2019-06-24)

[Full Changelog](https://github.com/kubeflow/training-operator/compare/v0.5.3...v1.0.0-rc.0)

# [v0.5.3](https://github.com/kubeflow/training-operator/tree/v0.5.3) (2019-06-03)

[Full Changelog](https://github.com/kubeflow/training-operator/compare/v0.5.2...v0.5.3)

# [v0.5.2](https://github.com/kubeflow/training-operator/tree/v0.5.2) (2019-05-23)

[Full Changelog](https://github.com/kubeflow/training-operator/compare/v0.5.1...v0.5.2)

# [v0.5.1](https://github.com/kubeflow/training-operator/tree/v0.5.1) (2019-05-15)

[Full Changelog](https://github.com/kubeflow/training-operator/compare/v0.5.0...v0.5.1)

# [v0.5.0](https://github.com/kubeflow/training-operator/tree/v0.5.0) (2019-03-26)

[Full Changelog](https://github.com/kubeflow/training-operator/compare/v0.4.0...v0.5.0)

# [v0.4.0](https://github.com/kubeflow/training-operator/tree/v0.4.0) (2019-02-13)

[Full Changelog](https://github.com/kubeflow/training-operator/compare/v0.4.0-rc.1...v0.4.0)

# [v0.4.0-rc.1](https://github.com/kubeflow/training-operator/tree/v0.4.0-rc.1) (2018-11-28)

[Full Changelog](https://github.com/kubeflow/training-operator/compare/v0.4.0-rc.0...v0.4.0-rc.1)

# [v0.4.0-rc.0](https://github.com/kubeflow/training-operator/tree/v0.4.0-rc.0) (2018-11-19)

[Full Changelog](https://github.com/kubeflow/training-operator/compare/v0.3.0...v0.4.0-rc.0)

# [v0.3.0](https://github.com/kubeflow/training-operator/tree/v0.3.0) (2018-09-22)

[Full Changelog](https://github.com/kubeflow/training-operator/compare/v0.2.0-rc1...v0.3.0)

# [v0.2.0-rc1](https://github.com/kubeflow/training-operator/tree/v0.2.0-rc1) (2018-06-21)

[Full Changelog](https://github.com/kubeflow/training-operator/compare/v0.1.0...v0.2.0-rc1)

# [v0.1.0](https://github.com/kubeflow/training-operator/tree/v0.1.0) (2018-03-29)

[Full Changelog](https://github.com/kubeflow/training-operator/compare/5b1ff9c7058c2af718ed8d399aebcfd124217f8c...v0.1.0)
