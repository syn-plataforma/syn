/* eslint-disable @typescript-eslint/naming-convention */
export class IncidenciaRegistrada {
  constructor(
    public description?: string,
    public bug_id?: number,
    public product?: string,
    public bug_Severity?: string,
    public priority?: string,
    public component?: string
  ) {}

  transform() {
    return {
      product: this.product,
      bug_severity: this.bug_Severity,
      priority: this.priority,
      component: this.component,
      bug_id: this.bug_id,
      description: this.description,
    };
  }
}
