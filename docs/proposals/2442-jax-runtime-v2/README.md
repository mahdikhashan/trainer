# KEP-2442: JAX Runtimes for Trainer V2

- [Summary](#summary)
- [Motivation](#motivation)
  - [Goals](#goals)
  - [Non-Goals](#non-goals)
- [Proposal](#proposal)
  - [User Stories (Optional)](#user-stories-optional)
    - [Story 1](#story-1)
    - [Story 2](#story-2)
- [Design Details](#design-details)
- [Alternatives](#alternatives)
- [Implementation History](#implementation-history)

## Summary

This proposal implements key components of KEP-2170, introducing the Kubeflow Training V2 API. Specifically, it focuses on creating the TrainingRuntime and ClusterTrainingRuntime for the JAX framework, built upon the Kubernetes JobSet API. These runtimes will serve as blueprints for model training (including LLMs) within cloud-native ML pipelines. This abstraction allows Data Scientists and MLOps Engineers to easily reuse standardized runtimes and launch training jobs, particularly via the SDK, without needing deep knowledge of underlying Kubernetes complexities.

## Motivation

JAX is a powerful Computation library created by Google, It is widely used in machine learning research and ranks as the third most wide used deep learning framework. JAX is more than a deep learning framework while suggests its potential in 
differential programming, large-scale physics simulations and many more. 

### Goals

- Implement ClusterTrainingRuntime for JAX, supporting single-node and multi-node configurations
- Build the necessary Docker images for JAX worker nodes used by the runtimes
- Document user guides for utilizing JAX TrainingRuntimes
- Test the implementation thoroughly using unit tests and end-to-end (E2E) tests

### Non-Goals

- GPU support (due to lack of available GPU testing infrastructure)
- Complex end-to-end examples demonstrating the runtimes (focus is on the runtime implementation itself; examples may require specific infrastructure)

## Proposal

<!--
This is where we get down to the specifics of what the proposal actually is.
This should have enough detail that reviewers can understand exactly what
you're proposing, but should not include things like API designs or
implementation. What is the desired outcome and how do we measure success?.
The "Design Details" section below is for the real
nitty-gritty.
-->

### User Stories (Optional)

#### Story 1

As a MLOps Engineer, I want to manage JAX distributed training jobs using the Kubeflow Trainer V2

#### Story 2


## Design Details

<!--
This section should contain enough information that the specifics of your
change are understandable. This may include API specs (though not always
required) or even code snippets. If there's any ambiguity about HOW your
proposal will be implemented, this is the place to discuss them.
-->

### Test Plan

<!--
The goal is to ensure that we don't accept enhancements with inadequate testing.
All code is expected to have adequate tests (eventually with coverage
expectations). Please adhere to the Kubeflow testing guidelines when drafting this test plan.
-->

[ ] I/we understand the owners of the involved components may require updates to
existing tests to make this code solid enough prior to committing the changes necessary
to implement this enhancement.

#### Unit Tests

<!--
In principle every added code should have complete unit test coverage, so providing
the exact set of tests will not bring additional value.
However, if complete unit test coverage is not possible, explain the reason of it
together with explanation why this is acceptable.
-->

<!--
Additionally, try to enumerate the core package you will be touching
to implement this enhancement and provide the current unit coverage for those
in the form of:
- <package>: <date> - <current test coverage>
This can inform certain test coverage improvements that we want to do before
extending the production code to implement this enhancement.
-->

- `<package>`: `<date>` - `<test coverage>`

#### E2E tests

<!--
Describe what E2E tests will be added to ensure proper quality of the enhancement.
After the implementation PR is merged, add the names of the tests here.
-->

#### Integration tests

<!--
Describe what tests will be added to ensure proper quality of the enhancement.
After the implementation PR is merged, add the names of the tests here.
-->

## Implementation History

- 2025-05-28: Initial KEP draft created.

## Drawbacks

<!--
Why should this KEP _not_ be implemented?
-->

## Alternatives

<!--
What other approaches did you consider, and why did you rule them out? These do
not need to be as detailed as the proposal, but should include enough
information to express the idea and why it was not acceptable.
-->