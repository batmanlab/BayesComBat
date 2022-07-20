import numpy as np
import numpyro
import jax
import jax.numpy as jnp
from jax import random
import numpyro.distributions as dist
from numpyro.distributions import constraints
import pandas as pd
from numpyro.infer import MCMC, NUTS
from numpyro.handlers import reparam
from numpyro.infer.reparam import LocScaleReparam
import os
import pickle



print('numpyro', numpyro.__version__)
print('jax', jax.__version__)
print('devices', jax.devices())


def infer(df, features, covariates, batch_var, subject_var, outdir):
    """Learns the ComBat model
    Arguments:
        df: pandas dataframe with all data (unharmonized imaging features, covariates, scanner indicators)
        features: list of feature names corresponding to df columns
        covariates: list of covariates corresponding to df columns. If categorical (i.e. Male/Female), please convert to int or float first
        batch_var: string column name corresponding to the scanner/site to harmonize
        subject: string column name corresponding to the subject identifier
        outdir: directory to save pickle files to
    """
    numpyro.enable_x64()

    if not os.path.exists(outdir):
        print("Making directory",outdir)
        os.mkdir(outdir)

    #normalize features
    for f in features:
        df[f]=(df[f] - df[f].mean()) / (df[f].std()) 

    # batch_var = 'Scanner_Proxy'
    # covariates = ['Age','Male','MCI','AD','Age*MCI','Age*AD']
    
    V = len(features)
    print(V, 'features')
    R = df.shape[0]
    print(R, 'measurements')

    #these should be 1's,1's,0's
    y_variances = jnp.array(df[features].var())
    y_stds = jnp.array(df[features].std())
    y_means = jnp.array(df[features].mean())

    feature_zeros=jnp.zeros(len(features),)
    feature_ones=jnp.ones(len(features),)
    feature_0_5=feature_ones*0.5

    unique_j = df[subject_var].unique() #unique subject id's
    unique_i = df[batch_var].unique() #unique scanners
    i_counts = jnp.array([sum(df[batch_var] == u_i) for u_i in unique_i]).reshape(-1,1) #counts of images from each scanners
    v_count = len(features)
    x_count = len(covariates)
    j_count = len(unique_j)
    i_count = len(unique_i)
    j_num = [np.argwhere(unique_j == df[subject_var][o])[0][0] for o in range(df.shape[0])] #unque subj num for each image
    i_num = [np.argwhere(unique_i == df[batch_var][o])[0][0] for o in range(df.shape[0])] #unique scanner num for each image

        
    def model(X_df, i, j, y_df=None):

        ###added for predictive
        j_num = [np.argwhere(unique_j == j.iloc[o])[0][0] for o in range(j.shape[0])] #unque subj num for each image
        i_num = [np.argwhere(unique_i == i.iloc[o])[0][0] for o in range(i.shape[0])] #unique scanner num for each image
        if y_df is not None:
            y = jnp.array(y_df.values)
        else:
            y=None
        X = jnp.array(X_df.values)
        ###

        alphas = numpyro.sample('alphas', dist.Independent(dist.Cauchy(feature_zeros, (feature_0_5)), 1))

        betas = numpyro.sample('betas', dist.ImproperUniform(constraints.real, (), (v_count, x_count)))

        rho = numpyro.sample('rho', dist.Independent(dist.InverseGamma(jnp.ones((v_count,))*6, jnp.ones((v_count,))*5), 1))

        with numpyro.plate("eta_plate", j_count):
            etas=numpyro.sample('etas', dist.Independent(dist.Normal(0,rho), 1))

        #fixed
        with numpyro.plate('gamma_high_level',i_count):
            gamma_i = numpyro.sample('gamma_i',dist.Cauchy(0,0.1))
            tau_i = numpyro.sample('tau_i', dist.InverseGamma(2,0.5))
            with numpyro.plate('gamma_plate', v_count):
                gammas_0 = numpyro.sample('gammas_0', dist.Normal(gamma_i, tau_i)) #gammas before transform
        gammas_0=gammas_0.T


        #fixed
        with numpyro.plate('delta_high_level', i_count):
            m_i=numpyro.sample('m_i', dist.Gamma(50,50)) #scanner-level scale
            s_i=numpyro.sample('s_i', dist.Gamma(50,1)) #scanner-level shape
            with numpyro.plate('delta_low_level', v_count):
                deltas_0_raw = numpyro.sample('deltas_0_raw', dist.Gamma(s_i,s_i))
                deltas_0 = numpyro.deterministic('deltas_0', deltas_0_raw*m_i) #deltas before transform


        #sigma_v
        sigmas_0 = numpyro.sample('sigmas_0', dist.Independent(dist.HalfCauchy(jnp.ones(v_count)*0.2),1))
        sigmas_0 = jnp.expand_dims(sigmas_0, 1)

        #new: get sigmas2 and deltas2 for transform
        sigmas2_0=numpyro.deterministic('sigmas2_0', sigmas_0**2)
        deltas2_0=numpyro.deterministic('deltas2_0', deltas_0**2)


        s = jnp.ones(len(unique_i)).reshape(-1,1)
        c = ((-jnp.matmul(i_counts.T, gammas_0))/(jnp.matmul(i_counts.T,s))).T
        gammas = numpyro.deterministic('gammas', gammas_0 + jnp.matmul(s,c.T))
        
        # pdb.set_trace()                  

        n=i_counts/(i_counts.sum())
        s=jnp.ones((i_count,1)) #vector of ones for transform
        # pdb.set_trace()
        sigmas2 = numpyro.deterministic('sigmas2',(jnp.matmul(deltas2_0,n)*jnp.matmul((jnp.matmul(sigmas2_0,s.T)),n))/(jnp.matmul(s.T,n))) 
        deltas2 = numpyro.deterministic('deltas2',((deltas2_0*jnp.matmul(sigmas2_0,s.T))/(jnp.matmul(jnp.matmul(deltas2_0,n)*jnp.matmul(sigmas2_0,jnp.matmul(s.T,n))/jnp.matmul(s.T,n),s.T))).T)

        sigmas = numpyro.deterministic('sigmas', jnp.sqrt(sigmas2))
        deltas = numpyro.deterministic('deltas', jnp.sqrt(deltas2))

        # pdb.set_trace()
        

        mu = alphas + jnp.matmul(X, betas.T) + etas[j_num,] + gammas[i_num,]

        with numpyro.plate('observations_plate', X.shape[0], dim=-2):
            numpyro.sample('obs', dist.Normal(mu, deltas[i_num,]*jnp.tile(sigmas,(1,y.shape[0])).T), obs = y)
            

    reparam_model = reparam(model, config={"etas": LocScaleReparam(0),
    "gammas_0":LocScaleReparam(0)})

    #Numpyro With MCMC

    rng_key = random.PRNGKey(0)
    rng_key, rng_key_ = random.split(rng_key)

    #10,000 samples but do 1000 x 10
    num_warmup, num_samples, n_iterations, warmup_thinning = 4000, 1000, 10, 10

    # Run NUTS.
    kernel = NUTS(reparam_model)
    num_chains = 4
    mcmc = MCMC(kernel, num_warmup = num_warmup, num_samples = num_samples, num_chains = num_chains, thinning=warmup_thinning)
    mcmc.warmup(rng_key_, X_df = df[covariates], i = df[batch_var], j = df[subject_var], y_df = df[features], collect_warmup=True)
    # warmup_samples=mcmc.get_samples(group_by_chain=True)

    with open(os.path.join(outdir,"mcmc_warmup.pickle"),"wb") as f:
        pickle.dump(mcmc, f, protocol = 4)

    mcmc.thinning=1
    mcmc.run(rng_key_, X_df = df[covariates], i = df[batch_var], j = df[subject_var], y_df = df[features])

    try:
        with open(os.path.join(outdir, 'mcmc_{}.pickle'.format("0")),"wb") as f:
            pickle.dump(mcmc, f, protocol = 4)
    except:
        print('couldnt write mcmc object')

    for i in range(1,n_iterations):
        mcmc.post_warmup_state = mcmc.last_state
        mcmc.run(mcmc.post_warmup_state.rng_key, X_df = df[covariates], i = df[batch_var], j = df[subject_var], y_df = df[features])
        try:
            with open(os.path.join(outdir, 'mcmc_{}.pickle'.format(i)),"wb") as f:
                pickle.dump(mcmc, f, protocol = 4)
        except:
            print('couldnt write mcmc object')

    print('inference complete')




def harmonize(df, features, covariates,batch_var, subject_var, outdir):
    """ Does haronization using pickle files created by infer
    Arguments:
        df: pandas dataframe with all data (unharmonized imaging features, covariates, scanner indicators).
        features: list of feature names corresponding to df columns
        covariates: list of covariates corresponding to df columns. If categorical (i.e. Male/Female), please convert to int or float first
        batch_var: string column name corresponding to the scanner/site to harmonize
        subject_var: string column name corresponding to the subject identifier
        outdir: directory to save pickle files to
    """


    df_unscaled=df.copy()
    #######
    for f in features:
        df[f]=(df[f] - df[f].mean()) / (df[f].std()) 
    #######



    V = len(features)
    print(V, 'features')
    R = df.shape[0]
    print(R, 'measurements')

    #these should be 1's,1's,0's
    y_variances = jnp.array(df[features].var())
    y_stds = jnp.array(df[features].std())
    y_means = jnp.array(df[features].mean())

    y_variances = jnp.array(df_unscaled[features].var())
    y_stds = jnp.array(df_unscaled[features].std())
    y_means = jnp.array(df_unscaled[features].mean())

    feature_zeros=jnp.zeros(len(features),)
    feature_ones=jnp.ones(len(features),)
    feature_0_5=feature_ones*0.5

    unique_j = df[subject_var].unique()
    unique_i = df[batch_var].unique()
    i_counts = jnp.array([sum(df[batch_var] == u_i) for u_i in unique_i]).reshape(-1,1)
    v_count = len(features)
    x_count = len(covariates)
    j_count = len(unique_j)
    i_count = len(unique_i)
    j_num = [np.argwhere(unique_j == df[subject_var][o])[0][0] for o in range(df.shape[0])]
    i_num = [np.argwhere(unique_i == df[batch_var][o])[0][0] for o in range(df.shape[0])] 


    print('Defining model')
    def model(X_df, i, j, y_df):
        
        y = jnp.array(y_df.values)
        X = jnp.array(X_df.values)
        

        alphas = numpyro.sample('alphas', dist.Independent(dist.Cauchy(feature_zeros, (feature_0_5)), 1))

        betas = numpyro.sample('betas', dist.ImproperUniform(constraints.real, (), (v_count, x_count)))

        #118 * 1
        rho = numpyro.sample('rho', dist.Independent(dist.InverseGamma(jnp.ones((v_count,))*6, jnp.ones((v_count,))*5), 1))

        with numpyro.plate("eta_plate", j_count):
            etas=numpyro.sample('etas', dist.Independent(dist.Normal(0,rho), 1))

        #fixed
        with numpyro.plate('gamma_high_level',i_count):
            gamma_i = numpyro.sample('gamma_i',dist.Cauchy(0,0.1))
            tau_i = numpyro.sample('tau_i', dist.InverseGamma(2,0.5))
            with numpyro.plate('gamma_plate', v_count):
                gammas_0 = numpyro.sample('gammas_0', dist.Normal(gamma_i, tau_i))
        gammas_0=gammas_0.T


        #fixed
        with numpyro.plate('delta_high_level', i_count):
            m_i=numpyro.sample('m_i', dist.Gamma(50,50)) #scanner-level scale
            s_i=numpyro.sample('s_i', dist.Gamma(50,1)) #scanner-level shape
            with numpyro.plate('delta_low_level', v_count):
                deltas_0_raw = numpyro.sample('deltas_0_raw', dist.Gamma(s_i,s_i))
                deltas_0 = numpyro.deterministic('deltas_0', deltas_0_raw*m_i)


        #sigma_v
        sigmas_0 = numpyro.sample('sigmas_0', dist.Independent(dist.HalfCauchy(jnp.ones(v_count)*0.2),1))
        sigmas_0 = jnp.expand_dims(sigmas_0, 1)

        #new: get sigmas2 and deltas2 for transform
        sigmas2_0=numpyro.deterministic('sigmas2_0', sigmas_0**2)
        deltas2_0=numpyro.deterministic('deltas2_0', deltas_0**2)


        s = jnp.ones(len(unique_i)).reshape(-1,1)
        c = ((-jnp.matmul(i_counts.T, gammas_0))/(jnp.matmul(i_counts.T,s))).T
        gammas = numpyro.deterministic('gammas', gammas_0 + jnp.matmul(s,c.T))
        n=i_counts/(i_counts.sum())
        s=jnp.ones((i_count,1)) #vector of ones for transform
        sigmas2 = numpyro.deterministic('sigmas2',(jnp.matmul(deltas2_0,n)*jnp.matmul((jnp.matmul(sigmas2_0,s.T)),n))/(jnp.matmul(s.T,n))) 
        deltas2 = numpyro.deterministic('deltas2',((deltas2_0*jnp.matmul(sigmas2_0,s.T))/(jnp.matmul(jnp.matmul(deltas2_0,n)*jnp.matmul(sigmas2_0,jnp.matmul(s.T,n))/jnp.matmul(s.T,n),s.T))).T)
        sigmas = numpyro.deterministic('sigmas', jnp.sqrt(sigmas2))
        deltas = numpyro.deterministic('deltas', jnp.sqrt(deltas2))
        mu = alphas + jnp.matmul(X, betas.T) + etas[j_num,] + gammas[i_num,]

        with numpyro.plate('observations_plate', X.shape[0], dim=-2):
            numpyro.sample('obs', dist.Normal(mu, deltas[i_num,]*jnp.tile(sigmas,(1,y.shape[0])).T), obs = y)
    reparam_model = reparam(model, config={"etas": LocScaleReparam(0),
    "gammas_0":LocScaleReparam(0)})



    n_results_files=10

    # experiment_paths={
    #     'current':'/jet/home/mare398/Pyro_Long_Combat/pickles/06_07_yt_ib_nd_ns_save_warmup_'}
    # save_path="/jet/home/mare398/Pyro_Long_Combat/evalfiles/06_07_yt_ib_nd_ns_save_warmup/"

    experiment_paths={
        'current':outdir}
    save_path=outdir

    experiment_samples={}
    experiment_samples_cat={}#concatenated samples frome each experiment
    for k in experiment_paths.keys():
        experiment_samples[k]=[]
    for k in experiment_paths:
        print(k)
        for i in range(n_results_files):
            print(experiment_paths[k]+str(i))
            with open(os.path.join(outdir,'mcmc_{}.pickle'),'rb') as f:
            # with open(experiment_paths[k]+str(i)+'.pickle', 'rb') as f:
                print('\tfile',i)
                mcmc = pd.read_pickle(f)
                # pdb.set_trace()
                experiment_samples[k].append(mcmc.get_samples())
                a_shape=experiment_samples['current'][i]['alphas'].shape
                alphas=experiment_samples['current'][i]['alphas']*y_stds.tile((a_shape[0],1))+y_means.tile((a_shape[0],1))
                b_shape=experiment_samples['current'][i]['betas'].shape
                betas=experiment_samples['current'][i]['betas']*jnp.repeat(jnp.expand_dims(y_stds.tile((b_shape[0],1)),2),6,2)#+jnp.repeat(jnp.expand_dims(y_means.tile((b_shape[0],b_shape[1],1)),3),6,3)
                g_shape=experiment_samples['current'][i]['gammas'].shape
                gammas=experiment_samples['current'][i]['gammas']*y_stds.tile((g_shape[0],g_shape[1],1))#+y_means.tile((g_shape[0],g_shape[1],g_shape[2],1))
                d_shape=experiment_samples['current'][i]['deltas'].shape
                deltas=experiment_samples['current'][i]['deltas']#*y_stds.tile((d_shape[0],d_shape[1],1))#+y_means.tile((d_shape[0],d_shape[1],d_shape[2],1))
                e_shape=experiment_samples['current'][i]['etas'].shape
                etas=experiment_samples['current'][i]['etas']*y_stds.tile((e_shape[0],e_shape[1],1))#+y_means.tile((e_shape[0],e_shape[1],e_shape[2],1))
                s_shape=experiment_samples['current'][i]['sigmas'].shape
                sigmas=(experiment_samples['current'][i]['sigmas']).reshape(s_shape[0],s_shape[1])**y_stds.tile((s_shape[0],1))

                n_samples=alphas.shape[0]
                y_ijv=jnp.expand_dims(df_unscaled[features].values,0).tile((n_samples,1,1))
                print('y_ijv',y_ijv.shape)
                try:
                    # with open(save_path+"y_ijv_{}.pickle".format(i),"wb") as f:
                    with open(os.path.join(save_path,"y_ijv_{}.pickle".format(i), "wb")) as f:
                        pickle.dump(y_ijv, f, protocol = 4)
                except:
                    print('couldnt write mcmc object')
                
                a_v=jnp.expand_dims(alphas,1).repeat(y_ijv.shape[1],1)
                print('a_v',a_v.shape)
                try:
                    with open(os.path.join(save_path,"a_v_{}.pickle".format(i)), "wb") as f:
                        pickle.dump(a_v, f, protocol = 4)
                except:
                    print('couldnt write mcmc object')

                b_v=betas
                print('b_v',b_v.shape)
                try:
                    with open(os.path.join(save_path,"b_v_{}.pickle".format(i)), "wb") as f:
                        pickle.dump(b_v, f, protocol = 4)
                except:
                    print('couldnt write mcmc object')

                x=jnp.array(df[covariates].values)
                print('x',x.shape)
                try:
                    with open(os.path.join(save_path,"x.pickle"), "wb") as f:
                        pickle.dump(x, f, protocol = 4)
                except:
                    print('couldnt write mcmc object')

                h_jv=etas[:,j_num,:]
                print('h_jv',h_jv.shape)
                try:
                    with open(os.path.join(save_path,"h_jv_{}.pickle".format(i)), "wb") as f:
                        pickle.dump(h_jv, f, protocol = 4)
                except:
                    print('couldnt write mcmc object')

                g_iv=gammas[:,i_num,:]
                print('g_iv',g_iv.shape)
                try:
                    with open(os.path.join(save_path,"g_iv_{}.pickle".format(i)), "wb") as f:
                        pickle.dump(g_iv, f, protocol = 4)
                except:
                    print('couldnt write mcmc object')

                d_iv=deltas[:,i_num,:]
                print('d_iv',d_iv.shape)
                try:
                    with open(os.path.join(save_path,"d_iv_{}.pickle".format(i)), "wb") as f:
                        pickle.dump(d_iv, f, protocol = 4)
                except:
                    print('couldnt write mcmc object')
                

                s_v=jnp.expand_dims(sigmas,1).repeat(y_ijv.shape[1],1)
                print('s_v',s_v.shape)
                try:
                    with open(os.path.join(save_path,"s_v_{}.pickle".format(i)), "wb") as f:
                        pickle.dump(s_v, f, protocol = 4)
                except:
                    print('couldnt write mcmc object')

                b_x_prod = jnp.array([jnp.matmul(x,b_v[i,:,:].T) for i in range(b_v.shape[0])])
                print('b_x_prod',b_x_prod.shape)

                try:
                    with open(os.path.join(save_path,"b_x_prod_{}.pickle".format(i)), "wb") as f:
                        pickle.dump(b_x_prod, f, protocol = 4)
                except:
                    print('couldnt write mcmc object')
                # pdb.set_trace()
                y_ijv_combat = (y_ijv-a_v-b_x_prod-h_jv-g_iv)/(d_iv)+a_v+b_x_prod+h_jv
                try:
                    with open(os.path.join(save_path,"y_ijv_harmonized_{}.pickle".format(i)), "wb") as f:
                        pickle.dump(y_ijv_combat, f, protocol = 4)
                except:
                    print('couldnt write mcmc object')


def load_harmonized_data(dir):
    """Load harmonized data. note: infer and harmonized must be run before this
    Arguments:
        dir: directory where harmonized samples are
    """

    num_pickles=10
    y_ijv_combat = None
    for p in range(num_pickles):
        print("Loading pickle #",p, end=" ")
        with open(os.path.join(dir,"y_ijv_combat_{}.pickle".format(p)),"rb") as f:
            y_ijv_combat_p=pickle.load(f)
            if y_ijv_combat is None:
                y_ijv_combat=y_ijv_combat_p
            else:
                y_ijv_combat=jnp.concatenate((y_ijv_combat,y_ijv_combat_p))
            print(y_ijv_combat.shape)
    print(y_ijv_combat.shape)
    return y_ijv_combat
