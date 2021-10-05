import { MyErrorTypes } from './error.enum';

export class MyError {
  constructor(
    private title: string,
    private description: string,
    private errorType: MyErrorTypes
  ) {}

  show() {
    switch (this.errorType) {
      case 0:
        console.log(this.toString());
        break;
      case 1:
        console.log(
          `%cWARN %c${this.toString()}`,
          'color:yellow',
          'color:gray'
        );
        break;
      case 2:
        console.log(`%cERROR %c${this.toString()}`, 'color:red', 'color:gray');
        break;
    }
  }
  private toString() {
    return `${this.title}: ${this.description}`;
  }
}
