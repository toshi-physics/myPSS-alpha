from .field import Field
import numpy as np
from .fourierfunc import *
from .explicitTerms import Term

class System:
    def __init__(self, grid_size, fourier_operators):
        """
        Initialize the system without any fields, but with a specified grid size.
        Parameters:
        grid_size -- the size of the grid, given as (num_rows, num_columns)
        """
        self.RK_index = 2
        self.atol = 1e-4
        self.rtol = 1e-4
        self.maxstep = 1e-2
        self.grid_size = grid_size
        self.fields = {}
        self.aux_fields = {}
        self.fourier_operators = fourier_operators 


    def create_field(self, field_name, k_list, k_grids, dynamic=False, dealias_factor=2/3):
        """
        Create a new field and additional RK_index fields for RK_index_th order Runge Kutta solver. Right now RK_index=2.
        Parameters:
        field_name -- the name of the field
        dynamic -- boolean indicating if the field is dynamic
        """
        if (field_name, 0) in self.fields:
            raise ValueError(f"A field with name {field_name} already exists.")
        
        self.fields[field_name] = Field(self.grid_size, k_list, k_grids, dynamic, dealias_factor)
        self.aux_fields[field_name] = Field(self.grid_size, k_list, k_grids, dynamic, dealias_factor)

    def get_field(self, field_name, is_aux=False):
        """
        Get the field object with the given name.
        Parameters:
        field_name -- the name of the field
        Returns:
        The Field object.
        """
        ret = self.fields[field_name]
        if is_aux:
            ret = self.aux_fields[field_name]
        return ret

    def create_term(self, field_name, fields, exponents):
        """
        Create a term in the update rule for the field with the given name.
        Parameters:
        field_name -- the name of the field
        fields -- a list of field names
        powers -- a list of powers for the spatial derivatives in Fourier space
        """
        assert field_name in self.fields, f"No field named '{field_name}' in the system."
        term = Term(fields, exponents)
        self.fields[field_name].add_term(term)
        self.aux_fields[field_name].add_term(term)
            
    def get_rhs_field(self, field_name, is_aux):
        """
        Update the field with the given name based on its terms.
        Parameters:
        field_name -- the name of the field
        """
        assert (field_name) in self.fields, f"No field named '{field_name}' in the system."
        field = self.fields[field_name]
        if is_aux:
            field = self.aux_fields[field_name]

        rhs_hat_total = np.zeros(field.data_momentum.shape, dtype=complex)     

        # term by term evaluation in real space then to k space then adding to total rhs in k space
        for term in field.get_terms():
            rhs = np.ones(field.data_real.shape)  
            for term_field_name, function_set in term.fields:
                rhs_field = self.get_field(term_field_name, is_aux)
                if function_set:
                    #print(function_set, field_name, term_field_name)
                    function, args = function_set
                    rhs *= function(rhs_field.get_real(), args)
                else:
                    rhs *= rhs_field.get_real()
                    
            rhs_hat = to_momentum_space(rhs)

            # if there are any fourier multiplications, do it. Else bypass.
            if np.sum(term.exponents[1:]) != 0: 
                for i, fourier_operator in enumerate(self.fourier_operators):
                    rhs_hat *= np.power(fourier_operator, term.exponents[i+1])

            rhs_hat *= term.exponents[0]
            
            rhs_hat_total += rhs_hat            
            
        return rhs_hat_total
            
    def update_system(self, dt):
        """

        """

        rhs_hats = {} #store rhs_hats for only dynamic fields, used to determine errors and next timesteps        
        #calculate all y_i's and k_i's, store latter into rhs_hats
                
        for field_name in self.aux_fields:
            field = self.aux_fields[field_name]
            if field.is_dynamic:
                rhs_hats[field_name, 1] = self.get_rhs_field(field_name, is_aux=False)
                field.add_to_momentum(dt*rhs_hats[field_name, 1])
            else:
                rhs_hat = self.get_rhs_field(field_name, is_aux=False)
                field.set_momentum(rhs_hat)

        for field_name in self.aux_fields:
            self.aux_fields[field_name].dealias_field()
            self.aux_fields[field_name].synchronize_real()
        
        # calculate final value of solution
        for field_name in self.fields:

            field = self.fields[field_name]
            if field.is_dynamic:
                #for dynamic fields follow Heun's Method
                rhs_hats[field_name, 2] = self.get_rhs_field(field_name, is_aux=True)
                field.add_to_momentum(dt*0.5*(rhs_hats[field_name, 1]+rhs_hats[field_name, 2]))
            else:
                #for static fields (which get updated after the dynamic fields) just calculate the value using the new y_i+1
                #this scheme depends on storing (and hence declaring) dynamic fields before static fields
                #later on this needs to be codified better
                rhs_hat = self.get_rhs_field(field_name, is_aux=False)
                field.set_momentum(rhs_hat)

        dt_next = 1e-6

        for field_name in self.fields:
            self.fields[field_name].dealias_field()
            self.fields[field_name].synchronize_real()

            # calculate the next time step based on the maximum relative error (over all space, over all dynamic fields)
            if self.fields[field_name].is_dynamic:
                absolute_error = (self.fields[field_name].get_real() - self.aux_fields[field_name].get_real() )
                tolerance = self.atol + self.rtol*np.maximum(np.abs(self.fields[field_name].get_real()), np.abs(self.aux_fields[field_name].get_real()))
                norm_error = absolute_error/tolerance
                dt_next = np.maximum(dt_next, dt * np.sqrt(1/np.max(norm_error)))

        dt_next= np.minimum(self.maxstep, dt_next)
        return dt_next
        
        

            
            

