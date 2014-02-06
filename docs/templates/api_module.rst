{{ name }}
{{ underline }}

.. automodule:: {{ fullname }}

   {% block functions %}
   {% if functions %}
Functions
=========
  
   .. container:: panel-group
   	:name: function_parent
   	
   	{% for item in functions %}
   	.. container:: panel panel-default
   		
   		.. container:: panel-heading
   			
   			.. raw:: html
   				
   				<a data-toggle="collapse" data-parent="#function_parent" href="#{{ item|replace("_", "-") }}">{{ item }}</a>
   		
   		.. container:: panel-collapse collapse
   			:name: {{ item|replace("_", "-") }}
   			
   			.. container:: panel-body
   				
   				.. autofunction:: {{ item }}
   	
   	
	{%- endfor %}

   {% endif %}
   {% endblock %}

   {% block classes %}
   {% if classes %}
Classes
=======

   .. autosummary::
    :toctree: api_generated/
    :template: api_class.rst
    
   {% for item in classes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block exceptions %}
   {% if exceptions %}
Exceptions
==========

.. autosummary::
   
   {% for item in exceptions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
