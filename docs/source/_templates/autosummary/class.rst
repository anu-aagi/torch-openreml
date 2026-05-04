{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}
   {% set special = ["__call__"] | select("in", members) | list %}
   {% set filtered_methods = methods | reject("equalto", "__init__") | list %}
   {% set display_methods = special + filtered_methods %}
   {% if display_methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
   {% for item in display_methods %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}