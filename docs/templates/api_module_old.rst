{{ name }}
{{ underline }}

.. automodule:: {{ fullname }}

   {% block functions %}
   {% if functions %}
Functions
=========

   .. autosummary::
	:nosignatures:
	
   {% for item in functions %}
      {{ item }}
   {%- endfor %}
   {% for item in functions %}
   .. container:: panel-collapse collapse in
   	:name: {{ item }}
   	
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
