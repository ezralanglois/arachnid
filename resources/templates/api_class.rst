{{ fullname }}
{{ underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}
   {% if methods %}

   .. autosummary::
	:nosignatures:
	
   {% for item in methods %}
      {{ item }}
   {%- endfor %}
   {% for item in methods %}
   .. automethod:: {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}