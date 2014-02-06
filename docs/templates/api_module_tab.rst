{{ name }}
{{ underline }}

.. automodule:: {{ fullname }}

   {% block functions %}
   {% if functions %}
Functions
=========

.. rst-class:: nav nav-tabs nav-stacked
   	
   	{% for item in functions %}
   		- `{{ item }} <#{{ item|replace("_", "-") }}>`_
	{%- endfor %}
	
   	.. container:: tab-content
   	
   	{% for item in functions %}
   		.. container:: tab-pane
   			:name: {{ item|replace("_", "-") }}
   		
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
