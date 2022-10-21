import numpy as np
import pandas as pd
from numpy import linalg as LA
from tqdm import tqdm


def hyper_sphere_coordindates(n_search_samples, instance, high, low, p_norm=2):
    delta_instance = np.random.randn(n_search_samples, instance.shape[1])
    dist = np.random.rand(n_search_samples) * (high - low) + low  # length range [l, h)
    norm_p = LA.norm(delta_instance, ord=p_norm, axis=1)
    d_norm = np.divide(dist, norm_p).reshape(-1, 1)  # rescale/normalize factor
    delta_instance = np.multiply(delta_instance, d_norm)
    candidate_counterfactuals = instance + delta_instance

    return candidate_counterfactuals, dist


def growing_spheres_search(
        data_name,
        instance,
        keys_mutable,
        keys_immutable,
        continuous_cols,
        binary_cols,
        feature_order,
        model,
        n_search_samples=4000,
        p_norm=2,
        step=0.2,
        max_iter=4000,
):
    keys_correct = feature_order

    keys_mutable_continuous = list(set(keys_mutable) - set(binary_cols))
    keys_mutable_binary = list(set(keys_mutable) - set(continuous_cols))

    instance_immutable_replicated = np.repeat(
        instance[keys_immutable].values.reshape(1, -1), n_search_samples, axis=0
    )
    instance_replicated = np.repeat(
        instance.values.reshape(1, -1), n_search_samples, axis=0
    )
    instance_mutable_replicated_continuous = np.repeat(
        instance[keys_mutable_continuous].values.reshape(1, -1),
        n_search_samples,
        axis=0,
    )

    low = 0
    high = low + step

    count = 0
    counter_step = 1

    # if data_name == 'adult':
    #     instance_ = instance.values.reshape(1, -1)
    #     age = instance_[:,0]
    #     hour_per_week = instance_[:,1]
    #     education = np.argmax(instance_[:,2:7], axis = 1)
    #     marital_status = np.argmax(instance_[:,7:11], axis = 1)
    #     occupation = np.argmax(instance_[:,11:], axis = 1)
    #     intance_value = np.hstack([education, marital_status, occupation, age, hour_per_week])
    #     instance_label = np.argmax(model.predict_proba(intance_value.reshape(1, -1)))
    # else:
    #     instance_label = np.argmax(model.predict_proba(instance.values.reshape(1, -1)))

    if data_name == 'adult':
        instance_ = instance.values.reshape(1, -1)
        age = instance_[:, 0]
        hour_per_week = instance_[:, 1]
        education = np.argmax(instance_[:, 2:10], axis=1)
        marital_status = np.argmax(instance_[:, 10:16], axis=1)
        occupation = np.argmax(instance_[:, 16:], axis=1)
        intance_value = np.hstack([education, marital_status, occupation, age, hour_per_week])
        instance_label = np.argmax(model.predict_proba(intance_value.reshape(1, -1)))
    else:
        instance_label = np.argmax(model.predict_proba(instance.values.reshape(1, -1)))

    #
    # instance_label = np.argmax(model.predict_proba(instance.values.reshape(1, -1)))
    # print(instance_label)

    counterfactuals_found = False
    candidate_counterfactual_star = np.empty(
        instance_replicated.shape[1],
    )
    candidate_counterfactual_star[:] = np.nan

    for _ in tqdm(range(max_iter)):
        if counterfactuals_found:
            break 
    # while not counterfactuals_found or count < max_iter:

        count = count + counter_step

        # STEP 1 -- SAMPLE POINTS on hyper sphere around instance
        candidate_counterfactuals_continuous, _ = hyper_sphere_coordindates(
            n_search_samples, instance_mutable_replicated_continuous, high, low, p_norm
        )

        # sample random points from Bernoulli distribution
        candidate_counterfactuals_binary = np.random.binomial(
            n=1, p=0.5, size=n_search_samples * len(keys_mutable_binary)
        ).reshape(n_search_samples, -1)

        # make sure inputs are in correct order
        candidate_counterfactuals = pd.DataFrame(
            np.c_[
                instance_immutable_replicated,
                candidate_counterfactuals_continuous,
                candidate_counterfactuals_binary,
            ]
        )
        candidate_counterfactuals.columns = (
            keys_immutable + keys_mutable_continuous + keys_mutable_binary
        )
        # enforce correct order
        candidate_counterfactuals = candidate_counterfactuals[keys_correct]

        # STEP 2 -- COMPUTE l_1 DISTANCES
        if p_norm == 1:
            distances = np.abs(
                (candidate_counterfactuals.values - instance_replicated)
            ).sum(axis=1)
        elif p_norm == 2:
            distances = np.square(
                (candidate_counterfactuals.values - instance_replicated)
            ).sum(axis=1)
        else:
            raise ValueError("Distance not defined yet")

        # pdb.set_trace()

        candidate_counterfactuals = candidate_counterfactuals.values

        if data_name == 'adult':
            age = candidate_counterfactuals[:, 0]
            hour_per_week = candidate_counterfactuals[:, 1]
            education = np.argmax(candidate_counterfactuals[:, 2:7], axis=1)
            marital_status = np.argmax(candidate_counterfactuals[:, 7:11], axis=1)
            occupation = np.argmax(candidate_counterfactuals[:, 11:], axis=1)
            candidate_counterfactuals = np.vstack([education, marital_status, occupation, age, hour_per_week]).T

        y_candidate_logits = model.predict_proba(candidate_counterfactuals)

        # y_candidate_logits = model.predict_proba(candidate_counterfactuals.values).reshape(-1)
        y_candidate = np.where(y_candidate_logits >= 0.5, 1, 0)
        indeces = np.where(y_candidate != instance_label)[0]
        candidate_counterfactuals = candidate_counterfactuals[indeces]
        candidate_dist = distances[indeces]

        # print(111111)
        # print(candidate_counterfactuals)

        if len(candidate_dist) > 0:  # certain candidates generated
            min_index = np.argmin(candidate_dist)
            candidate_counterfactual_star = candidate_counterfactuals[min_index]
            counterfactuals_found = True


        # print(22222)
        # print(candidate_counterfactual_star)

        # no candidate found & push search range outside
        low = high
        high = low + step
    print(candidate_counterfactual_star)
    return candidate_counterfactual_star